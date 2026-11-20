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
   preflight pulls the image on every docker node, then, with a weights plan
   (``--model-store-source``, :mod:`.ssh_weights`), every node copies the
   model (or its pipeline stages' files) into its local store at once. The
   node-local data (store stage, release staging, temp files, init release,
   ``~/yeto-output``) goes under the node's inventory ``scratch`` and the
   incarnation's state dir (:mod:`.ssh_node_storage`);
3. the SSH session streams node 0's log to the head (``IslandEvents`` reads the
   trainer events from it); the other nodes' logs go to local files;
4. release kills each node's process group (TERM, then KILL after a grace
   period) by the PID file the start wrote; a restarted head does the same for
   an orphan incarnation (the PID file path is derived from the run id, island
   name and incarnation, recorded in the journal before any start). Each
   incarnation is also a row of the run's resource ledger, so ``yeto cleanup``
   and the run's reaper stop it when the head is gone (:mod:`.ssh_trainer_ledger`);
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
Without the trainer scheduler the same island is a resident one: the fleet
starts it once and supervises it for the run (:mod:`.ssh_resident`). The code
reaches the nodes as the run code pack (downloaded from the run's s3 store), the
release image, or a checkout at ``--rl-ssh-trainer-workdir``; the image is the
run's ``--rl-image``. Nothing here was run against a real node.
"""

from __future__ import annotations

import base64
import ipaddress
import re
import shlex
import subprocess
import sys
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
from . import ssh_node_storage, ssh_trainer_ledger, ssh_transport
from .credentials import _secret_value
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


def _overlay_safe_image(image: str) -> str | None:
    """The overlay-safe repack of a pinned engine image (miles pins), or None."""
    from yeto.rl.engine import backends

    try:
        return backends.module("pins", "miles").overlay_safe_image(image)
    except Exception:  # noqa: BLE001 - a hint only: never mask the pull error itself
        return None


SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=4")
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
# The open-file limit of a trainer container: docker gives it dockerd's default (often
# 1024), and a multi-node Ray raylet runs out of sockets at that ("Too many open
# files"), its node is declared dead and the actors die. Added unless docker_args
# set their own --ulimit nofile.
NOFILE_ULIMIT = "nofile=1048576:1048576"
CUDA_RDMA_DOCKER_ARGS = ("--gpus", "all", "--device", "/dev/infiniband", "--ulimit", "memlock=-1:-1",
                         "--ulimit", "stack=67108864", "--cap-add", "IPC_LOCK")


class SshIslandError(RuntimeError):
    """The SSH trainer island cannot start or stop as asked."""


class _TransientSsh(SshIslandError):
    """An ``ssh`` the node dropped before the remote command ran (ssh_transport)."""


@dataclass(frozen=True)
class SshTrainerSpec:
    """What every incarnation runs.

    ``command``: the trainer entry, as ``torchrun`` arguments after ``--``
    (e.g. ``["-m", "glm_rl.verl_ext.train", ...]``); or ``run_script``: a bash
    script run on every node with the sky task variables (module doc); exactly
    one of the two. ``workdir``: the directory it runs in (inside the container
    on docker nodes); ``requant``: argv of the re-quantizer side process on
    node 0 (None: BF16 samplers only). ``image`` / ``docker_args`` replace the
    nodes' inventory values on docker nodes (e.g. a trainer image). ``files``: the
    island task's file mounts (``{path on the node: local path}``, a path under
    ``~`` as in a sky task), shipped inside the start script to every node: bind-
    mounted read-only into the container on docker nodes (``~`` is
    ``container_home`` there), copied under the job's ``$HOME`` on rootfs nodes.
    ``code_pack``: ``(uri, digest)`` of the run code pack in the run's s3 store; every
    node downloads and unpacks it (:data:`PACK_FETCH_PY`) and links ``~/sky_workdir``
    to it, the tree the island checks against ``--code-pack-digest`` (``workdir`` may
    then be empty: the container starts in ``container_home``, a rootfs job in ``$HOME``)."""

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
    files: Mapping[str, str] = field(default_factory=dict)
    container_home: str = "/root"
    code_pack: tuple[str, str] | None = None  # (s3:// URI, digest) of the run code pack
    weights: Any = None  # ssh_weights.WeightsPlan: the preflight copies the model onto every node

    def __post_init__(self) -> None:
        if not self.nodes:
            raise SshIslandError("an SSH trainer island needs at least one node")
        if bool(self.command) == bool(self.run_script):
            raise SshIslandError("an SSH trainer island needs one trainer command or one run script")
        if not self.workdir and self.code_pack is None:
            raise SshIslandError("an SSH trainer island needs a workdir (the yeto checkout on the nodes) or a "
                                 "run code pack")
        if self.code_pack is not None and not (str(self.code_pack[0]).startswith("s3://")
                                               and re.fullmatch(r"[0-9a-f]{64}|<[a-z-]+>", self.code_pack[1])):
            raise SshIslandError(f"code pack {self.code_pack!r}: expected an s3:// URI and a sha256 digest")
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
        for target in self.files:
            if not (str(target).startswith(("~/", "/")) and ".." not in Path(str(target)).parts):
                raise SshIslandError(f"file mount {target!r}: the node path must be absolute or under ~/")
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


# Runs on every node (in the container on docker nodes) before the island: download
# the run code pack from the run's s3 store with the island's store credentials, check
# its sha256 and unpack it with the pack's own unpacker (yeto.codepack.unpack: the tree
# the island then verifies against --code-pack-digest) into the codepacks cache. Kept
# free of yeto imports until the pack is there; boto3 is in every trainer image.
PACK_CACHE = "$HOME/.cache/yeto/codepacks"
PACK_FETCH_PY = r"""
import hashlib, io, os, shutil, sys, tarfile, tempfile
uri, digest, cache = sys.argv[1], sys.argv[2].lower(), os.path.expanduser(sys.argv[3])
if os.path.isfile(os.path.join(cache, digest, "CODEPACK.json")):
    sys.exit(0)
import boto3
from botocore.config import Config
env = os.environ
kw = {}
if env.get("YETO_RL_STORE_S3_ACCESS_KEY_ID") and env.get("YETO_RL_STORE_S3_SECRET_ACCESS_KEY"):
    kw = {"aws_access_key_id": env["YETO_RL_STORE_S3_ACCESS_KEY_ID"],
          "aws_secret_access_key": env["YETO_RL_STORE_S3_SECRET_ACCESS_KEY"],
          "aws_session_token": env.get("YETO_RL_STORE_S3_SESSION_TOKEN") or None}
client = boto3.session.Session(**kw).client(
    "s3", endpoint_url=env.get("YETO_RL_STORE_S3_ENDPOINT_URL") or None,
    region_name=env.get("YETO_RL_STORE_S3_REGION") or None,
    config=Config(retries={"mode": "standard", "max_attempts": 9}))
bucket, _, key = uri.split("://", 1)[1].partition("/")
data = client.get_object(Bucket=bucket, Key=key)["Body"].read()
if hashlib.sha256(data).hexdigest() != digest:
    sys.exit(f"[yeto-island] code pack {uri} does not have digest {digest}")
os.makedirs(cache, exist_ok=True)
tmp = tempfile.mkdtemp(prefix=".fetch-", dir=cache)
try:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith("yeto/") and m.isfile()
                   and not os.path.isabs(m.name) and ".." not in m.name.split("/")]
        tar.extractall(tmp, members=members)
    sys.path.insert(0, tmp)
    from yeto.codepack import unpack
    print(f"[yeto-island] code pack {digest[:12]} unpacked to {unpack(data, digest, cache)}", flush=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
"""


def _code_link(spec: SshTrainerSpec) -> str:
    """Bash that points ``~/sky_workdir`` at the node's code: the run code pack (fetched
    first) or the spec's workdir; nothing when neither (a release image links it)."""
    if spec.code_pack is not None:
        uri, digest = spec.code_pack
        tree = f"\"{PACK_CACHE}/{digest}\""
        return (f"python3 -I -c {shlex.quote(PACK_FETCH_PY)} {shlex.quote(uri)} {shlex.quote(digest)} "
                f"\"{PACK_CACHE}\" || {{ echo '[yeto-island] could not fetch the run code pack' >&2; exit 1; }}\n"
                'if [ -d "$HOME/sky_workdir" ] && [ ! -L "$HOME/sky_workdir" ]; then '
                "echo '[yeto-island] ~/sky_workdir is a directory: the run code pack cannot be linked there' >&2; "
                "exit 1; fi\n"
                f'ln -sfn {tree} "$HOME/sky_workdir"\n')
    if spec.workdir:
        return f'[ -e "$HOME/sky_workdir" ] || ln -s {shlex.quote(spec.workdir)} "$HOME/sky_workdir"\n'
    return ""


def _trainer_body(spec: SshTrainerSpec, rank: int, requant_log: str, state_dir: str | None = None) -> str:
    """The node's job as bash; it exits with the trainer's exit code. ``state_dir``:
    the incarnation's state dir as the job sees it (the node-local storage prelude,
    :func:`ssh_node_storage.job_prelude`)."""
    node = spec.nodes[rank]
    prelude = ssh_node_storage.job_prelude(node, spec.run_id, state_dir)
    if spec.run_script:
        # The island run script (rl_island_task) enters ~/sky_workdir, where sky syncs
        # the code; an SSH node has no sync: the run code pack, or the spec's workdir.
        trainer = f"{_code_link(spec)}(\n{spec.run_script.rstrip()}\n)"  # a subshell: exit and exec stay inside
    else:
        trainer = shlex.join([spec.torchrun, f"--nnodes={len(spec.nodes)}", f"--node-rank={rank}",
                              f"--nproc-per-node={node.cards_per_node}",
                              f"--master-addr={spec.addresses()[0]}",
                              f"--master-port={spec.master_port}", *spec.command])
    if rank == 0 and spec.requant:
        # The loop converts releases while the trainer runs; after the trainer
        # ends, one more pass converts the last BF16 release.
        return (f"{prelude}{shlex.join(spec.requant)} --loop --poll-seconds {spec.requant_poll_s:g} "
                f"> {requant_log} 2>&1 &\n"
                "req=$!\n"
                f"{trainer} && rc=0 || rc=$?\n"
                "kill $req 2>/dev/null; wait $req 2>/dev/null\n"
                f"{shlex.join(spec.requant)} >> {requant_log} 2>&1 || {{ [ $rc -ne 0 ] || rc=1; }}\n"
                "exit $rc\n")
    return f"{prelude}{trainer}\n"


def docker_argv(spec: SshTrainerSpec, rank: int, name: str, env_names: Sequence[str]) -> list[str]:
    """``docker run`` of one node, the image last; the environment goes by name
    (``-e KEY``), so no value appears in the process list."""
    node = spec.nodes[rank]
    argv = ["docker", "run", "--rm", "--name", name, "--network", "host", "--ipc", "host",
            *spec.docker_args_of(node)]
    if not any(f.startswith("--ulimit=nofile=") for f in _flag_forms(spec.docker_args_of(node))):
        argv += ["--ulimit", NOFILE_ULIMIT]
    for host_path, container_path in sorted(node.mounts.items()):
        argv += ["-v", f"{host_path}:{container_path}"]
    argv += ssh_node_storage.docker_options(node, spec.run_id)
    for key in sorted(env_names):
        argv += ["-e", key]
    return [*argv, "-w", spec.workdir or spec.container_home, str(spec.image or node.image)]


def node_env(spec: SshTrainerSpec, rank: int, env: Mapping[str, str]) -> dict[str, str]:
    """The environment of node ``rank``'s job: its family, the fabric defaults
    (``topology.island_comm_env``), the inventory's ``env`` (e.g. NCCL_IB_HCA refines
    the defaults), the spec's environment, ``env`` and the sky task variables."""
    node = spec.nodes[rank]
    comm = island_comm_env(node.device_family, node.fabric, node.interface, num_nodes=len(spec.nodes))
    # The learner reads the family to pick its card type and collective backend (accel).
    full = {DEVICE_FAMILY_ENV: node.device_family, **comm, **node.env, **spec.env, **env,
            **(skypilot_env(spec, rank) if spec.run_script else {})}
    # node-local data (store stage, release staging, temp files) under the node's scratch
    full.update(ssh_node_storage.scratch_env(node, spec.run_id, full))
    full.update(ssh_node_storage.transfer_env(node, full))  # a release goes up in processes
    return full


def docker_job(spec: SshTrainerSpec, rank: int, paths: Mapping[str, str], container: str,
               full_env: Mapping[str, str], inner_word: str) -> str:
    """The ``docker run`` command line of node ``rank`` (``paths`` pinned, ``$HOME``
    expands on the node; ``inner_word``: the quoted ``bash -c`` argument). The
    environment goes by name; the incarnation's state dir and the shipped files are
    bind-mounted."""
    *options, image = docker_argv(spec, rank, container, list(full_env))
    mount = f"-v \"{paths['dir']}:{CONTAINER_STATE_DIR}\""
    home = str(full_env.get("HOME") or spec.container_home)
    mount += "".join(f" -v \"{paths['dir']}/files/{i}:{_home_path(target, home)}:ro\""
                     for i, target in enumerate(sorted(spec.files)))
    return f"{shlex.join(options)} {mount} {shlex.quote(image)} bash -c {inner_word}"


def start_script(spec: SshTrainerSpec, rank: int, env: Mapping[str, str], paths: Mapping[str, str],
                 *, container: str | None = None) -> str:
    """The bash script one node runs: start the island in its own session, then
    stream its log until it exits (node 0's stream carries the trainer events).
    ``container``: the docker container name (docker nodes)."""
    paths = _pin_state_home(paths)
    node = spec.nodes[rank]
    full_env = node_env(spec, rank, env)
    files = _file_payloads(spec)
    if spec.docker:
        if container is None:
            raise SshIslandError("a docker node needs its container name")
        # The state dir is mounted so the re-quantizer log lands next to the trainer log
        # (and ~/yeto-output in it: the tape outlives the container).
        inner = _trainer_body(spec, rank, f"{CONTAINER_STATE_DIR}/requant.log", CONTAINER_STATE_DIR)
        job = docker_job(spec, rank, paths, container, full_env, shlex.quote(inner))
        enter = place = ""
    else:
        requant_log = f"\"{paths['requant_log']}\""
        job = f"bash -c {shlex.quote(_trainer_body(spec, rank, requant_log))}"
        enter = f"cd {shlex.quote(spec.workdir)}\n" if spec.workdir else 'cd "$HOME"\n'
        place = "".join(
            f"__yeto_f={_home_path(target, '$HOME', quote=False)}\n"
            f"mkdir -p \"$(dirname \"$__yeto_f\")\" && cp \"{paths['dir']}/files/{i}\" \"$__yeto_f\"\n"
            for i, target, _ in files)
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
        # owner-only: the logs and the tape of the job live there
        + f"(umask 077 && mkdir -p \"{paths['dir']}\")\n"
        + ssh_node_storage.host_prepare(node, spec.run_id)
        + _write_files(files, paths)
        + place
        + enter
        + f"rm -f \"{paths['rc']}\"\n"
        # Job control is off in a non-interactive shell, so the background setsid
        # is not a group leader: it does not fork, and $! is the new group's id.
        # The subshell exits at once, so init reaps the island when it ends (a
        # zombie would keep both ``tail --pid`` and the kill check waiting).
        + f"( setsid bash -c {shlex.quote(body)} > \"{paths['log']}\" 2>&1 < /dev/null &\n"
        + f"  echo $! > \"{paths['pid']}\" )\n"
        + _follow(paths)
    )


def _follow(paths: Mapping[str, str]) -> str:
    """Stream the incarnation's log from its start until its process group leader exits."""
    return f"exec tail -n +1 -F --pid=\"$(cat \"{paths['pid']}\")\" \"{paths['log']}\" 2>/dev/null\n"


def follow_script(paths: Mapping[str, str]) -> str:
    """Reattach to a running (or ended) incarnation: replay its log from the start and
    follow it until the island exits (a restarted head, or an SSH stream that dropped)."""
    return f"{STATE_HOME_VAR}=\"$HOME\"\n" + _follow(_pin_state_home(paths))


NODE_STATE_MARK = "yeto-node-state"
NODE_STATES = ("running", "exited", "lost", "absent")


@dataclass(frozen=True)
class NodeState:
    """One node of one incarnation, as :func:`status_script` reports it: ``running``
    (its process group lives), ``exited`` (its job wrote the exit code ``rc``),
    ``lost`` (the state dir exists but neither process nor exit code: killed,
    OOM-killed or the node rebooted) or ``absent`` (never started there).
    ``container``: the docker container status (``none`` when there is none)."""

    state: str
    rc: int | None = None
    container: str = "none"

    @property
    def running(self) -> bool:
        return self.state == "running"


def status_script(paths: Mapping[str, str], *, container: str | None = None) -> str:
    """One line ``yeto-node-state state=... rc=... container=...`` (:class:`NodeState`)."""
    p = _pin_state_home(paths)
    probe = (f"c=$(docker inspect -f '{{{{.State.Status}}}}' {shlex.quote(container)} 2>/dev/null) || c=none\n"
             if container else "c=none\n")
    return (
        f"{STATE_HOME_VAR}=\"$HOME\"\nrc=\n"
        f"if [ -f \"{p['rc']}\" ]; then s=exited; rc=$(tr -d '[:space:]' < \"{p['rc']}\" 2>/dev/null)\n"
        f"elif pid=$(cat \"{p['pid']}\" 2>/dev/null) && [ -n \"$pid\" ] && kill -0 -- -$pid 2>/dev/null; "
        "then s=running\n"
        f"elif [ -d \"{p['dir']}\" ]; then s=lost\n"
        "else s=absent\nfi\n"
        + probe
        + f"echo \"{NODE_STATE_MARK} state=$s rc=$rc container=${{c:-none}}\"\n"
    )


def parse_node_state(text: str) -> NodeState | None:
    """The :class:`NodeState` in ``text`` (a :func:`status_script` output); None without one."""
    for line in str(text or "").splitlines():
        if not line.startswith(NODE_STATE_MARK + " "):
            continue
        fields = dict(item.partition("=")[::2] for item in line.split()[1:])
        state = fields.get("state", "")
        if state not in NODE_STATES:
            return None
        rc = fields.get("rc") or ""
        return NodeState(state, int(rc) if re.fullmatch(r"-?\d+", rc) else None,
                         fields.get("container") or "none")
    return None


def _home_path(target: str, home: str, *, quote: bool = True) -> str:
    """``target`` with a leading ``~`` replaced by ``home``. ``quote=False``: a shell
    word in which ``home`` (e.g. ``$HOME``) still expands."""
    target = str(target)
    if not target.startswith("~"):
        return target if quote else shlex.quote(target)
    rest = target[1:]
    if quote:
        return f"{home}{rest}"
    return f"\"{home}\"{shlex.quote(rest) if rest else ''}"


def _file_payloads(spec: SshTrainerSpec) -> list[tuple[int, str, str]]:
    """``(index, node path, base64 content)`` of every file mount of the spec."""
    out = []
    for i, (target, local) in enumerate(sorted(spec.files.items())):
        try:
            data = Path(str(local)).expanduser().read_bytes()
        except OSError as error:
            raise SshIslandError(f"file mount {target!r}: cannot read {local!r} ({error})") from None
        out.append((i, str(target), base64.b64encode(data).decode("ascii")))
    return out


def _write_files(files: Sequence[tuple[int, str, str]], paths: Mapping[str, str]) -> str:
    """Bash that writes the shipped files into the incarnation's state dir (mode 600)."""
    if not files:
        return ""
    out = f"mkdir -p \"{paths['dir']}/files\"\n"
    for i, _target, payload in files:
        wrapped = "\n".join(payload[j:j + 76] for j in range(0, len(payload), 76))
        out += (f"( umask 077; base64 -d > \"{paths['dir']}/files/{i}\" <<'__YETO_FILE_{i}__'\n"
                f"{wrapped}\n__YETO_FILE_{i}__\n)\n")
    return out


def kill_script(paths: Mapping[str, str], grace_s: int = KILL_GRACE_S, *, container: str | None = None) -> str:
    """TERM the island's process group, KILL it after ``grace_s``; exit 0 when it
    is gone. ``container``: stop and remove that docker container first (a killed
    ``docker run`` client does not stop its container): TERM to its processes,
    ``docker rm -f`` after ``grace_s``; a container still there exits 4.

    The pid file is read from the login HOME, where :func:`start_script` pins the
    state dir. No pid file: a warning, and exit 0 only when the state dir does not
    exist either (this incarnation never started on the node); a state dir with no
    pid file exits 3, so a release cannot report ok while the island may still run."""
    remove = ""
    if container:
        c = shlex.quote(container)
        # TERM to every process of the container but its init (``docker stop`` signals
        # only PID 1, a bash that ignores TERM): the trainer gets the stop signal it
        # gets on Modal (a preemption marker, no save), then the container goes.
        remove = (
            f"if docker container inspect {c} >/dev/null 2>&1; then\n"
            f"  docker exec {c} bash -c 'kill -TERM -1' >/dev/null 2>&1 || true\n"
            f"  for _ in $(seq {int(grace_s)}); do "
            f"[ \"$(docker inspect -f '{{{{.State.Running}}}}' {c} 2>/dev/null)\" = true ] || break; sleep 1; done\n"
            f"  docker rm -f {c} >/dev/null 2>&1 || true\n"
            "fi\n"
            f"if docker container inspect {c} >/dev/null 2>&1; then\n"
            f"  echo \"[yeto] kill: container {container} still exists after docker rm -f\" >&2; exit 4\n"
            "fi\n"
        )
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
    running ``ssh`` process with its stdout as a text pipe. Every call reuses one
    master connection per node, and ``run`` / ``fetch`` retry a connection the node
    dropped before a session existed (:mod:`.ssh_transport`); a start streams its
    output and is sent once."""

    retry_delays_s: Sequence[float] = ssh_transport.RETRY_DELAYS_S
    sleep: Callable[[float], None] = staticmethod(time.sleep)

    def command(self, node: Node) -> list[str]:
        """``ssh`` to ``node`` running the script it reads from stdin: a start script
        exports the task's secrets, which must not be in an argv (``ps`` on either
        end, a ``TimeoutExpired`` message). The remote bash reads the whole script
        before it runs, so nothing in it (or in a login profile) can consume the rest
        of it."""
        route = [*node.ssh_options(), node.host]
        return ["ssh", *SSH_OPTIONS, *ssh_transport.mux_options(route), *route, "bash", "-c",
                shlex.quote(READ_SCRIPT)]

    def run(self, node: Node, script: str, *, timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
        return ssh_transport.run_with_retry(
            subprocess.run, self.command(node), delays=self.retry_delays_s, sleep=self.sleep,
            log=lambda line: print(line, flush=True), label=node.name,
            input=script, text=True, capture_output=True, timeout=timeout)

    def fetch(self, node: Node, script: str, target: Path, *, timeout: float) -> int:
        """Run ``script`` on ``node`` and extract the tar it writes to stdout under
        ``target`` (:func:`ssh_node_storage.extract_tar_stream`); the files written.
        Raises on a failed or timed-out copy; a dropped connection is retried."""
        for delay in (*self.retry_delays_s, None):
            try:
                return self._fetch_once(node, script, target, timeout=timeout)
            except _TransientSsh as error:
                if delay is None:
                    raise SshIslandError(str(error)) from None
                print(f"[launcher] ssh to {node.name}: {error}; retrying in {delay:g} s", flush=True)
                self.sleep(delay)
        raise AssertionError("unreachable")

    def _fetch_once(self, node: Node, script: str, target: Path, *, timeout: float) -> int:
        proc = subprocess.Popen(self.command(node), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        timer = threading.Timer(float(timeout), proc.kill)
        timer.start()
        try:
            proc.stdin.write(script.encode())
            proc.stdin.close()
            count = ssh_node_storage.extract_tar_stream(proc.stdout, target)
            err = proc.stderr.read().decode(errors="replace")
            rc = proc.wait()
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()
        if rc != 0:
            message = f"copy from {node.name} ended with rc {rc}: {err.strip()[-300:]}"
            if count == 0 and ssh_transport.transient_failure(rc, err):
                raise _TransientSsh(message)
            raise SshIslandError(message)
        return count

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
    attached: bool = False  # follows an incarnation an earlier head started


class SshTrainerIslands:
    """``TrainerIslands`` over SSH nodes (module doc).

    ``scheduled=False``: an island the fleet keeps up for the whole run
    (:mod:`.ssh_resident`), not one the trainer scheduler drives: its incarnations
    start without the scheduler's control environment (``island_env``), so the
    trainer runs its own rounds to the end. ``on_line``: every line of node 0's
    stream (the launcher's log and echo tape); ``tape``: write the scheduler's
    per-incarnation event tape into ``events_dir``."""

    def __init__(self, spec: SshTrainerSpec, *, name: str, jobs_path: Path, ops: SshOps | None = None,
                 poll_s: float | None = None, events_dir: Path | None = None,
                 clock: Callable[[], float] = time.time, kill_grace_s: int = KILL_GRACE_S,
                 scheduled: bool = True, tape: bool = True,
                 on_line: Callable[[str], None] | None = None,
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
        self.scheduled = bool(scheduled)
        self.tape = bool(tape)
        self.on_line = on_line
        self.node_wait_s = float(node_wait_s)
        self.node_poll_s = float(node_poll_s)
        self.sleep = sleep
        self.jobs: list[_SshJob] = []
        self.checked = False
        self.unreleased: set[int] = set()  # incarnations not confirmed stopped on every node

    # -- preflight ----------------------------------------------------------------
    def preflight(self) -> None:
        """Every node's CPU architecture and cards must match the inventory, and a
        docker node must pull the image (raises before any start). The nodes are
        checked in parallel (a first pull of a trainer image takes minutes per node)."""
        nodes = list(self.spec.nodes)
        if len(nodes) == 1:
            self._preflight_node(nodes[0])
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(16, len(nodes)), thread_name_prefix="yeto-preflight") as pool:
                for future in [pool.submit(self._preflight_node, node) for node in nodes]:
                    future.result()  # the first failed node in inventory order raises
        warning = ssh_node_storage.no_scratch_warning(
            list(self.spec.nodes), staged=bool(ssh_node_storage.staged_envs(self.spec.env)))
        if warning:
            print(warning, flush=True)
        if self.spec.weights is not None:
            self._sync_weights()
        self.checked = True

    def _preflight_node(self, node: Node) -> None:
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
        self._check_scratch(node)
        if self.spec.docker:
            self._pull(node)

    def _check_scratch(self, node: Node) -> None:
        """The node's scratch directory exists, is writable and has the free space
        the inventory asks for; its store env file is owner-only (ssh_node_storage)."""
        script = ssh_node_storage.env_file_preflight_script(node)
        if script is not None:
            done = self.ops.run(node, script, timeout=60.0)
            try:
                ssh_node_storage.check_env_file(node, done.returncode, done.stdout or "")
            except ssh_node_storage.ScratchError as error:
                raise SshIslandError(str(error)) from None
        script = ssh_node_storage.preflight_script(node, self.spec.run_id)
        if script is None:
            return
        done = self.ops.run(node, script, timeout=120.0)
        try:
            free = ssh_node_storage.check_preflight(node, done.returncode, done.stdout or "",
                                                    getattr(done, "stderr", "") or "")
        except ssh_node_storage.ScratchError as error:
            raise SshIslandError(str(error)) from None
        print(f"[launcher] node {node.name}: scratch {node.scratch} has {free:.0f} GiB free", flush=True)

    def _sync_weights(self) -> None:
        """Every node copies the model into its local store at once (ssh_weights)."""
        from .ssh_weights import WeightsSyncError, sync_weights

        try:
            sync_weights(self.spec.weights, self.spec.nodes, self.spec.addresses(), self.ops,
                         images=[str(self.spec.image or n.image) if self.spec.docker else None
                                 for n in self.spec.nodes])
        except WeightsSyncError as exc:
            raise SshIslandError(str(exc)) from None

    def _pull(self, node: Node) -> None:
        """Pull the trainer image before the first start: a first ``docker run`` of a
        24 GB image would spend the start timeout, and a layer the node's docker
        cannot unpack shows here, not as a dead incarnation."""
        image = str(self.spec.image or node.image)
        done = self.ops.run(node, f"docker pull -q {shlex.quote(image)}", timeout=PULL_TIMEOUT_S)
        if done.returncode != 0:
            err = ((getattr(done, "stderr", "") or "") + (getattr(done, "stdout", "") or "")).strip()[-600:]
            hint = ""
            if "not a directory" in err:
                fixed = _overlay_safe_image(image)
                hint = (f" (a layer that whites out a path and writes it again; the same image with that "
                        f"layer repacked: {fixed})" if fixed else
                        " (a layer that whites out a path and writes it again: rebuild it with "
                        "docker/miles-ports/make_layer.py of this repository)")
            raise SshIslandError(f"node {node.name}: docker pull {image} failed (rc {done.returncode}): {err}{hint}")

    # -- TrainerIslands -------------------------------------------------------------
    def launch(self, incarnation: int, resume_from: CutRef | None) -> _SshJob:
        if not self.checked:
            self.preflight()
        else:
            self._wait_reachable(incarnation)
        self._stop_unreleased(incarnation)
        paths = remote_paths(self.spec.run_id, self.name, incarnation)
        control = island_env(incarnation, resume_from, poll_s=self.poll_s) if self.scheduled else {}
        env = {**control, SUBMITTED_ENV: f"{self.clock():.3f}"}
        save_job(self.jobs_path, incarnation, f"ssh:{self.name}-i{int(incarnation)}")  # before any start
        # in the run's resource ledger: a cleanup (or the reaper) stops it without this head
        ssh_trainer_ledger.record("created", **self._ledger_key(incarnation), attrs=ssh_trainer_ledger.created_attrs(
            **self._ledger_key(incarnation), nodes=self.spec.nodes, docker=self.spec.docker,
            kill_grace_s=self.kill_grace_s))
        job = self._job(incarnation, paths)
        try:
            for rank, node in enumerate(self.spec.nodes):
                job.procs.append(self.ops.start(node, start_script(
                    self.spec, rank, env, paths, container=self._container(incarnation, rank))))
        except Exception:
            self.release(job)
            raise
        for rank in range(len(job.procs)):
            self._pump(job, rank)
        self.jobs.append(job)
        print(f"[launcher] trainer incarnation {incarnation} on {len(self.spec.nodes)} SSH node(s) "
              f"(resume {None if resume_from is None else resume_from.cut_id})", flush=True)
        return job

    def attach(self, incarnation: int) -> _SshJob:
        """Follow incarnation ``incarnation`` that an earlier head started (its log
        replays from the start, then streams until the island exits)."""
        paths = remote_paths(self.spec.run_id, self.name, incarnation)
        job = self._job(incarnation, paths)
        job.attached = True
        try:
            for node in self.spec.nodes:
                job.procs.append(self.ops.start(node, follow_script(paths)))
        except Exception:
            for proc in job.procs:
                if proc.poll() is None:
                    proc.terminate()
            raise
        for rank in range(len(job.procs)):
            self._pump(job, rank)
        self.jobs.append(job)
        self.checked = True  # the nodes run this island: no preflight before a later start
        print(f"[launcher] reattached to trainer incarnation {incarnation} on {len(self.spec.nodes)} "
              "SSH node(s)", flush=True)
        return job

    def follow(self, job: _SshJob, rank: int) -> Any:
        """A new stream of node ``rank`` (its previous SSH stream dropped while the
        island still runs there)."""
        old = job.procs[rank]
        if old.poll() is None:
            old.terminate()
        job.procs[rank] = self.ops.start(self.spec.nodes[rank], follow_script(job.paths))
        self._pump(job, rank)
        return job.procs[rank]

    def node_state(self, incarnation: int, rank: int, *, timeout: float = 60.0) -> NodeState | None:
        """What node ``rank`` reports for ``incarnation``; None when it does not answer."""
        script = status_script(remote_paths(self.spec.run_id, self.name, incarnation),
                               container=self._container(incarnation, rank))
        try:
            done = self.ops.run(self.spec.nodes[rank], script, timeout=timeout)
        except Exception:  # noqa: BLE001 - timeout / ssh missing: no answer
            return None
        return parse_node_state(getattr(done, "stdout", "") or "")

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
    def _job(self, incarnation: int, paths: dict[str, str]) -> _SshJob:
        tape = incarnation_tape(self.events_dir, self.name, incarnation) if self.tape else None
        return _SshJob(int(incarnation), paths, [], IslandEvents(tape))

    def _pump(self, job: _SshJob, rank: int) -> None:
        if rank == 0:
            on_line = self.on_line

            def sink(line: str) -> None:
                job.events.feed(line)
                if on_line is not None:
                    on_line(line)
        else:
            sink = self._node_log(job.incarnation, rank)
        thread = threading.Thread(target=_pump, args=(job.procs[rank], sink), daemon=True,
                                  name=f"{self.name}-i{job.incarnation}-n{rank}")
        thread.start()
        job.threads.append(thread)

    def _container(self, incarnation: int, rank: int) -> str | None:
        return container_name(self.spec.run_id, self.name, incarnation, rank) if self.spec.docker else None

    def _ledger_key(self, incarnation: int) -> dict[str, Any]:
        return {"run_id": self.spec.run_id, "island": self.name, "incarnation": int(incarnation)}

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
        ok = self._kill_nodes(incarnation)
        if ok:  # every node confirmed: the ledger row closes (else a cleanup retries it)
            ssh_trainer_ledger.record("released", **self._ledger_key(incarnation))
        self._fetch_outputs(incarnation)
        return ok

    def _fetch_outputs(self, incarnation: int) -> None:
        """Copy each node's ``~/yeto-output`` of this incarnation into
        ``<events dir>/<island>-i<N>-output/node<rank>/`` (ssh_node_storage); a failure
        is a warning, never a failed release. Nothing without an events dir."""
        if self.events_dir is None:
            return
        paths = _pin_state_home(remote_paths(self.spec.run_id, self.name, incarnation))
        base = Path(self.events_dir) / f"{self.name}-i{int(incarnation)}-output"
        for rank, node in enumerate(self.spec.nodes):
            remote = ssh_node_storage.output_dir(node, paths["dir"])
            script = f"{STATE_HOME_VAR}=\"$HOME\"\n" + ssh_node_storage.output_fetch_script(remote)
            try:
                count = self.ops.fetch(node, script, base / f"node{rank}",
                                       timeout=ssh_node_storage.OUTPUT_FETCH_TIMEOUT_S)
                if count:
                    print(f"[launcher] trainer incarnation {incarnation}: {count} output file(s) of "
                          f"{node.name} -> {base / f'node{rank}'}", flush=True)
            except Exception as error:  # noqa: BLE001 - the tape copy never fails the release
                print(f"[launcher] WARN: could not copy ~/yeto-output of trainer incarnation {incarnation} "
                      f"from {node.name} ({type(error).__name__}: {error}); it stays on the node",
                      flush=True)

    def _kill_nodes(self, incarnation: int) -> bool:
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


def run_image(args: Any, nodes: Sequence[Node]) -> str | None:
    """The trainer image of docker nodes: the run's ``--rl-image`` (the image its pins
    are checked against); an inventory image is optional and must name the same one.

    A pinned image whose layers an overlay applier refuses (moby overlay2, the
    containerd snapshotter of Docker 29; ``pins.OVERLAY_SAFE_DIGESTS``) runs as its
    overlay-safe repack: the same files and Miles commit, so the pins still hold. An
    inventory may name either digest."""
    ref = str(getattr(args, "rl_image", None) or "")
    ref = ref[len("docker:"):] if ref.startswith("docker:") else ref
    docker = [n for n in nodes if n.runtime == "docker"]
    if not docker:
        return None
    if not ref:
        images = {n.image for n in docker}
        only = next(iter(images)) if len(images) == 1 else None
        safe = _overlay_safe_image(only) if only else None
        if safe:
            print(f"[launcher] ssh trainer image {only} -> its overlay-safe repack {safe}", file=sys.stderr,
              flush=True)
        return safe  # None: the inventory's images (SshTrainerSpec refuses a node without one)
    safe = _overlay_safe_image(ref)
    other = sorted({n.image for n in docker if n.image and n.image not in (ref, safe)})
    if other:
        names = [n.name for n in docker if n.image and n.image not in (ref, safe)]
        raise SshIslandError(f"inventory image {other} of nodes {names} is not the run's --rl-image {ref}: the "
                             "run's image decides; drop the inventory image (or images:), or pass it as --rl-image")
    if safe:
        print(f"[launcher] ssh trainer image {ref} -> its overlay-safe repack {safe}", file=sys.stderr,
              flush=True)
        return safe
    return ref


CODE_PACK_SOURCE, RELEASE_SOURCE, WORKDIR_SOURCE = "code-pack", "release", "workdir"


def code_source(args: Any) -> str | None:
    """How the run's code reaches an ssh: island's nodes: the run code pack from the
    s3 store (``code-pack``), the release image (``release``), the checkout at
    ``--rl-ssh-trainer-workdir`` (``workdir``), or None (it does not)."""
    from .code_pack import CODE_PACK_GIT_PATCH, CODE_PACK_RELEASE, code_pack_mode

    mode = code_pack_mode(args) if getattr(args, "training_mode", "sft") == "rl" else None
    store = str(getattr(args, "rl_checkpoint_store", None) or "")
    if mode == CODE_PACK_RELEASE:
        return RELEASE_SOURCE
    if mode == CODE_PACK_GIT_PATCH and store.startswith("s3://"):
        return CODE_PACK_SOURCE
    return WORKDIR_SOURCE if getattr(args, "rl_ssh_trainer_workdir", None) else None


def island_code_pack(args: Any) -> tuple[str, str] | None:
    """``(uri, digest)`` of the run code pack the nodes download (None: no pack). A launch
    has stored it already (``prepare_run_code_pack``); a dry run only plans it."""
    if code_source(args) != CODE_PACK_SOURCE:
        return None
    pack = getattr(args, "code_pack", None)
    if pack is not None and pack.pack_uri:
        return str(pack.pack_uri), str(pack.digest)
    store = str(args.rl_checkpoint_store).rstrip("/")
    return f"{store}/code_packs/<digest>.tar", "<digest>"


def island_spec_from_inventory(args: Any, island: Any, task: Any) -> SshTrainerSpec:
    """The :class:`SshTrainerSpec` of the held-back ``ssh:`` trainer island: the
    inventory's trainer nodes run the island's sky task (setup, then run) as
    ``run_script`` with the task's environment. The code: the run code pack (every node
    downloads it from the store, :func:`code_source`), the release image, or the
    checkout at ``--rl-ssh-trainer-workdir`` (SSH nodes get no sky workdir sync). The
    image: ``--rl-image`` (:func:`run_image`). The task's file mounts (e.g. the expert
    selection) travel in the start script."""
    workdir = getattr(args, "rl_ssh_trainer_workdir", None) or ""
    code_pack = island_code_pack(args)
    if not workdir and code_pack is None and code_source(args) != "release":
        raise SshIslandError("an ssh: trainer island needs the run's code on its nodes: a run code pack (an "
                             "s3:// --rl-checkpoint-store with --rl-code-pack auto or git-patch: every node "
                             "downloads it), a release image (--rl-code-pack release), or "
                             "--rl-ssh-trainer-workdir (a yeto checkout already on the nodes)")
    run = getattr(task, "run", None)
    if not isinstance(run, str) or not run.strip():
        raise SshIslandError("the ssh: trainer island's task has no run script")
    setup = getattr(task, "setup", None) or ""
    env = {**(getattr(task, "envs", None) or {}), **(getattr(task, "secrets", None) or {})}
    files = {str(k): str(v) for k, v in (getattr(task, "file_mounts", None) or {}).items()
             if isinstance(v, (str, Path))}  # a bucket mount (dict / Storage) is no file
    remote = sorted(k for k, v in files.items() if "://" in v)
    if remote:
        raise SshIslandError(f"the ssh: trainer island's task mounts {remote} from a bucket; SSH nodes "
                             "get local files only")
    # sky.Task keeps secrets as pydantic SecretStr, whose str() is the '**********' mask
    env = {k: _secret_value(v) for k, v in env.items() if v is not None}
    nodes = inventory_trainer_nodes(args, island)
    from .ssh_weights import WeightsSyncError, weights_plan_from_args

    try:
        weights = weights_plan_from_args(args, env, nodes)
    except WeightsSyncError as exc:
        raise SshIslandError(str(exc)) from None
    return SshTrainerSpec(nodes=nodes, run_id=str(args.cluster_prefix), workdir=str(workdir),
                          image=run_image(args, nodes), code_pack=code_pack,
                          # sky runs setup and run as two scripts; a failed setup ends the job
                          run_script=f"bash -c {shlex.quote(setup)} || exit $?\n{run}" if setup.strip() else run,
                          env=env, files=files, weights=weights)
