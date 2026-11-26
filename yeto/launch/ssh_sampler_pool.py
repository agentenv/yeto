"""Sampler pool on the operator's own SSH nodes (pool config ``provider.kind: ssh``).

The National Compute, Daytona, Modal and sky providers rent machines; a site
with its own nodes (e.g. Ascend 910B) had no provider, so its samplers were
started by hand. This provider takes the nodes of an SSH node inventory
(:mod:`yeto.launch.node_inventory`, the trainer's ``--rl-node-inventory``
format) as its slots: slot ``i`` is ``provider.nodes[i]``. Nothing is ordered or
billed: ``order`` reserves a ticket in a local record, ``release`` stops the
sampler and marks the ticket ended. Only a pool config that names
``provider.kind: ssh`` reaches this module; every other provider is unchanged.

What runs on a node, for each start (all idempotent):

* preflight: the card tool of the node's family (``nvidia-smi``, ``amd-smi`` /
  ``rocm-smi``, ``npu-smi info``; the inventory's card check), the number of cards
  against the inventory, optionally no other process on the cards
  (``require_idle``: the process table of ``npu-smi info``, on by default for an
  ascend node; the compute apps of ``nvidia-smi`` / ``ppu-smi`` and ``amd-smi
  process`` when the config sets ``require_idle: true``), the host devices a rocm
  container needs (``/dev/kfd``, ``/dev/dri``), and the free space of ``workdir``
  (``min_free_gib``); a failed check refuses the start before anything runs;
* the yeto tree of the launcher's commit (``git archive``) under
  ``<workdir>/yeto/<sha>``, the member's rendered spec under
  ``<workdir>/sampler/<shard>.json``;
* the sampler, detached, either in a container (``runtime: container``: podman or
  docker) or as a process of a venv on the host (``runtime: venv``: no container
  runtime): by default the node runtime (``python -m yeto.node run``,
  :mod:`yeto.node.route`) starts it, and its ready record, heartbeats and exit go
  to the run store; ``node_runtime: false`` (in ``provider`` or the pool config)
  starts ``python -m yeto.rl.engine.engine_sampler --spec`` directly, as before;
* its log in ``<workdir>/logs/<shard>.log`` (container: ``<engine> logs``),
  read back by :meth:`SshSamplerPoolOps.logs`; its event tape next to the spec.

Stop is by PID, never by name matching: the start script writes the PID of the
sampler process before it ``exec``s python; ``stop`` checks with ``ps -o args=``
that this PID is still the engine sampler, sends it SIGTERM (the sampler runs
with ``YETO_SPOT_RECLAIM_MARKER`` set, so SIGTERM is its orderly leave: it stops
the engine and exits), waits ``grace_s``, and only then kills what is left of
the process group of that start. A container is stopped with ``<engine> stop -t
grace_s`` (SIGTERM to its PID 1, the sampler).

Device defaults by family (:class:`~yeto.launch.node_inventory.NodeProfile`) for a
node without ``docker_args``: cuda ``--gpus`` (docker) or CDI ``nvidia.com/gpu=<i>``
(podman); rocm the NC route's ``ROCM_DOCKER_ARGS`` (``/dev/kfd``, ``/dev/dri``, the video
group, seccomp off, ...; the pool's ``container.shm_size`` stays the one shm size);
ascend ``/dev/davinci<i>`` plus the host devices and driver mounts the preflight found.
The visible-card variable (``CUDA_VISIBLE_DEVICES`` as the container numbers the
cards, ``HIP_VISIBLE_DEVICES``, ``ASCEND_RT_VISIBLE_DEVICES``) is set only when
``devices`` is; a cuda / rocm / ppu node also gets ``YETO_DEVICE_FAMILY`` (torch reads
ROCm and PPU as ``cuda``). A node with ``docker_args`` and an ascend node start
exactly as before (no added flag or variable).

``container.image: "auto"`` takes each node's image from the engine image catalog
(:func:`yeto.launch.engine_images.resolve` with what the sampler spec asks: model,
engine, reload mode, precision, rollout fields). ``pool --dry-run`` refuses a node no
image serves, and a node whose family or card the spec's ``platform`` /
``engine.server.device_family`` / ``card_model`` name otherwise.

Container runtime knobs (what an old podman on Ascend needs, A02 of hetero-1114):
``security_opt`` (``["seccomp=unconfined"]``: the default seccomp profile of
podman 0.10 blocks ``statx`` and every CANN operator fails as if the versions did
not match), ``pull_ref`` (pull by the arm64 digest: an old podman cannot read a
manifest list) and ``image`` (what ``run`` names; an image id after a digest
pull). Nothing of that is fixed in code.

Data path: the pool's ``data_path`` (``r2`` with ``r2_env_file``, or
``syncer``) as on the other routes, or ``provider.store``: a store directory on
the node (spec ``store`` / ``buffer_root``). A node-local store is only read on
that node, so the pool's fence cannot reach it: ``store`` needs ``target: 1``,
and a fence there is the sampler stop.

Spare nodes: ``slots`` may list more nodes than ``target``. When a node is lost under
its sampler (the pool reads it unreachable for ``elastic.unreachable_grace_s``), the
pool fences the shard and orders a replacement that continues its prompts; the lost
node's slot is held for ``provider.lost_node_hold_s`` (default 1800 s; 0: no hold) while
any other slot's node is free, so the replacement starts on a spare, not on the lost
node. A stop the lost node never got runs before the next start there
(:class:`SshNodeProvider`). On the store control plane (``rollout.control_plane: store``)
the tickets, holds and owed stops live in the run store
(``fleet/pools/<pool>/provider/<token>.json``, one record per lock holder), so a
controller that takes the pool over on another host keeps them.

Reach: every ``ssh`` to a node (preflight, code upload, start, probe, stop) goes
through the node's ``reach`` in the inventory (:func:`yeto.node.transport.reach_options`):
``direct``; ``jump`` (``ProxyJump``; ``proxy_jump:`` is the same); ``relay`` (``via:
nc:<account>``: a National Compute workspace relay; ``via: wss://...``: a ``relay.mjs``
the site runs behind an authenticating gateway, its bearer token in
``$YETO_RELAY_TOKEN_FILE`` / ``$YETO_RELAY_TOKEN``: :mod:`yeto.node.relay_proxy`; a relay
node's host key is checked strictly). An ``outbound`` / ``function`` node is refused
(no ssh reaches it).

Pool config::

    "provider": {"kind": "ssh", "inventory": "~/ascend-nodes.yaml", "nodes": ["node-a"],
                 "workdir": "/srv/yeto/run1", "runtime": "container",
                 "container": {"engine": "podman", "image": "3a7d901460ab",
                               "pull_ref": "quay.nju.edu.cn/ascend/vllm-ascend@sha256:<arm64>",
                               "pull": "missing", "security_opt": ["seccomp=unconfined"]},
                 "env": {"ASCEND_RT_VISIBLE_DEVICES": "0"}, "devices": [0],
                 "store": "/srv/yeto/run1/store",
                 "preflight": {"min_free_gib": 20, "require_idle": true}}
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import engine_images, ssh_transport
from .node_inventory import (
    NPU_HOST_DEVICES,  # noqa: F401 - re-exported (they lived here)
    NPU_HOST_MOUNTS,  # noqa: F401
    Node,
    NodeProfile,
    load_profiles,
    nodes_by_name,
    profile_of,
)
from .pool import status as island_status
from .pool.provider import provider_caps
from .pool.status import LeaseStatus
from yeto.node.store import STORE_URI_SHELL, pool_fence_store, reclaim_marker_env, sampler_route_env
from yeto.node.transport import ReachError, node_ssh_options

KIND = "ssh"
SSH_UNREACHABLE = 255  # ssh's own failure (no connection, no session): the node was not reached
REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "yeto-ssh-sampler"  # prefix of a start's marker (start_marker): run, shard, ticket
MARK_ENV = "YETO_SSH_SAMPLER_MARK"  # the sampler process carries its start's marker in this variable
RUNTIMES = ("container", "venv")
ENGINES = ("podman", "docker")
PULL_POLICIES = ("missing", "always", "never")
CONFIG_KEYS = frozenset({"kind", "inventory", "nodes", "workdir", "runtime", "container", "venv", "env",
                         "devices", "store", "preflight", "grace_s", "price_usd_h", "start_timeout_s",
                         "lost_node_hold_s", "node_runtime"})
CONTAINER_KEYS = frozenset({"engine", "image", "pull_ref", "pull", "security_opt", "sudo", "args", "python",
                            "shm_size", "mounts"})
VENV_KEYS = frozenset({"python", "source", "ld_preload", "pythonpath"})
PREFLIGHT_KEYS = frozenset({"min_free_gib", "require_idle", "min_cards"})
AUTO_IMAGE = "auto"  # container.image: each node's image from the engine image catalog
# ``node_runtime`` unset: the node runtime starts the sampler, as on the sky route;
# ``node_runtime: false`` keeps the sampler start of before
NODE_RUNTIME_DEFAULT = True
# NPU_HOST_DEVICES / NPU_HOST_MOUNTS (what an Ascend container needs from the host besides
# the cards) now live in node_inventory with the other per-family defaults; imported above.


class SshSamplerError(RuntimeError):
    """A start that cannot go on (preflight refused, the launch did not come up)."""


class PreflightError(SshSamplerError):
    config_error = False  # a busy card or a full disk is this node's, not the config's


# -- config ---------------------------------------------------------------------


@dataclass(frozen=True)
class ContainerConf:
    engine: str = "podman"
    image: str | None = None
    pull_ref: str | None = None
    pull: str = "missing"
    security_opt: tuple[str, ...] = ()
    sudo: bool = False
    args: tuple[str, ...] = ()
    python: str = "python3"
    shm_size: str = "16g"
    mounts: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VenvConf:
    python: str
    source: tuple[str, ...] = ()  # env scripts sourced in order (CANN set_env.sh, ATB set_env.sh)
    ld_preload: str | None = None
    pythonpath: tuple[str, ...] = ()  # appended after the yeto tree, before the inherited PYTHONPATH


@dataclass(frozen=True)
class SshPoolConf:
    nodes: tuple[NodeProfile, ...]
    workdir: str
    runtime: str
    container: ContainerConf | None
    venv: VenvConf | None
    env: Mapping[str, str]
    devices: tuple[int, ...] | None
    store: str | None
    min_free_gib: float
    require_idle: bool | None  # None: the default (on for an ascend node, off for the others)
    min_cards: int | None
    grace_s: int
    price_usd_h: float | None
    # a node lost under a sampler (unreachable, its stop not confirmed) is left out this long
    # while another slot's node is free: its slot moves to a spare node (unified-sampler-pool 3.3);
    # 0: no hold (the lost node's slot is offered at once)
    lost_node_hold_s: float = 1800.0
    # the node starts ``python -m yeto.node run`` (yeto.node.route), which starts the sampler
    # (the default); false: the sampler directly, as before (kept for one release)
    node_runtime: bool = True


def _strs(value: Any, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"ssh provider: {key} must be a list of strings, not {value!r}")
    return tuple(value)


def _unknown(where: str, section: Mapping[str, Any], known: frozenset[str]) -> None:
    extra = set(section) - known
    if extra:
        raise ValueError(f"ssh provider: unknown keys {sorted(extra)} in {where} (known: {sorted(known)})")


def parse_config(cfg: Mapping[str, Any], *, inventory: Sequence[Node] | None = None) -> SshPoolConf:
    """The provider section of a pool config; any gap refuses the pool before SSH."""
    section = dict(cfg.get("provider") or {})
    _unknown("provider", section, CONFIG_KEYS)
    if inventory is None:
        if not section.get("inventory"):
            raise ValueError("ssh provider: inventory (an SSH node inventory file) is required")
        inventory = load_profiles(os.path.expanduser(str(section["inventory"])))
    inventory = [profile_of(n) for n in inventory]
    by_name = nodes_by_name(list(inventory))
    names = section.get("nodes")
    if names is None:
        nodes = [n for n in inventory if not n.roles or "sampler" in n.roles]
    else:
        missing = [n for n in _strs(names, "nodes") if n not in by_name]
        if missing:
            raise ValueError(f"ssh provider: nodes {missing} are not in the inventory ({sorted(by_name)})")
        nodes = [by_name[n] for n in names]
    if not nodes:
        raise ValueError("ssh provider: no sampler node (nodes, or inventory nodes with role sampler)")
    unreachable = [f"{n.name} ({n.reach.mode})" for n in nodes if n.reach is not None and n.reach.mode in
                   ("outbound", "function")]
    if unreachable:
        raise ValueError(f"ssh provider: nodes {unreachable} take no SSH connection; this pool starts samplers "
                         "over SSH")
    for node in nodes:  # every other route must be dialable before anything runs (jump / relay: token, TLS)
        if node.reach is None or node.reach.mode not in ("outbound", "function"):
            try:
                node_ssh_options(node)
            except ReachError as error:
                raise ValueError(f"ssh provider: node {node.name}: {error}") from None
    workdir = str(section.get("workdir") or "")
    if not workdir.startswith("/") or workdir.rstrip("/") in ("", "/tmp", "/data", "/root", "/home"):
        raise ValueError(f"ssh provider: workdir {workdir!r} must be an absolute directory of this run")
    runtime = str(section.get("runtime") or "")
    if runtime not in RUNTIMES:
        raise ValueError(f"ssh provider: runtime {runtime!r} must be one of {RUNTIMES}")
    container = venv = None
    if runtime == "container":
        raw = dict(section.get("container") or {})
        _unknown("container", raw, CONTAINER_KEYS)
        engine = str(raw.get("engine", "podman"))
        if engine not in ENGINES:
            raise ValueError(f"ssh provider: container.engine {engine!r} must be one of {ENGINES}")
        pull = str(raw.get("pull", "missing"))
        if pull not in PULL_POLICIES:
            raise ValueError(f"ssh provider: container.pull {pull!r} must be one of {PULL_POLICIES}")
        image = raw.get("image") or raw.get("pull_ref")
        if image == AUTO_IMAGE and raw.get("pull_ref"):
            raise ValueError("ssh provider: container.image auto pulls the catalog image itself; drop pull_ref")
        if not image and not all(n.image for n in nodes):
            raise ValueError("ssh provider: container.image (or pull_ref, or the inventory's image) is required")
        mounts = raw.get("mounts") or {}
        if not isinstance(mounts, dict):
            raise ValueError("ssh provider: container.mounts must be a map of host path to container path")
        container = ContainerConf(engine=engine, image=str(image) if image else None,
                                  pull_ref=str(raw["pull_ref"]) if raw.get("pull_ref") else None, pull=pull,
                                  security_opt=_strs(raw.get("security_opt"), "container.security_opt"),
                                  sudo=bool(raw.get("sudo", False)), args=_strs(raw.get("args"), "container.args"),
                                  python=str(raw.get("python", "python3")), shm_size=str(raw.get("shm_size", "16g")),
                                  mounts=dict(mounts))
    else:
        raw = dict(section.get("venv") or {})
        _unknown("venv", raw, VENV_KEYS)
        if not str(raw.get("python") or "").startswith("/"):
            raise ValueError("ssh provider: venv.python (absolute path of the venv's python) is required")
        venv = VenvConf(python=str(raw["python"]), source=_strs(raw.get("source"), "venv.source"),
                        ld_preload=str(raw["ld_preload"]) if raw.get("ld_preload") else None,
                        pythonpath=_strs(raw.get("pythonpath"), "venv.pythonpath"))
    env = section.get("env") or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
        raise ValueError("ssh provider: env must be a map of name to value")
    devices = section.get("devices")
    if devices is not None:
        if not isinstance(devices, list) or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0
                                                    for d in devices) or not devices:
            raise ValueError(f"ssh provider: devices must be a non-empty list of card indexes, not {devices!r}")
        devices = tuple(devices)
    store = section.get("store")
    if store is not None and not str(store).startswith("/"):
        raise ValueError(f"ssh provider: store {store!r} must be an absolute directory on the node")
    pre = dict(section.get("preflight") or {})
    _unknown("preflight", pre, PREFLIGHT_KEYS)
    price = section.get("price_usd_h")
    hold = section.get("lost_node_hold_s", 1800)
    if isinstance(hold, bool) or not isinstance(hold, (int, float)) or hold < 0:
        raise ValueError(f"ssh provider: lost_node_hold_s must be a number of seconds >= 0, not {hold!r}")
    from yeto.node.route import enabled as node_runtime_enabled  # noqa: PLC0415

    try:  # provider.node_runtime, else the pool config's (default true; false: the sampler start of before)
        node_runtime = node_runtime_enabled(cfg, default=NODE_RUNTIME_DEFAULT, section=section)
    except ValueError as error:
        raise ValueError(f"ssh provider: {error}") from None
    return SshPoolConf(nodes=tuple(nodes), workdir=workdir.rstrip("/"), runtime=runtime, container=container,
                       venv=venv, env={str(k): str(v) for k, v in env.items()}, devices=devices,
                       store=str(store) if store else None, min_free_gib=float(pre.get("min_free_gib", 20)),
                       require_idle=(bool(pre["require_idle"]) if pre.get("require_idle") is not None else None),
                       min_cards=int(pre["min_cards"]) if pre.get("min_cards") is not None else None,
                       grace_s=int(section.get("grace_s", 120)),
                       price_usd_h=float(price) if price is not None else None, lost_node_hold_s=float(hold),
                       node_runtime=node_runtime)


# -- node shell -----------------------------------------------------------------


def node_paths(workdir: str, shard: str) -> dict[str, str]:
    s = f"{workdir}/sampler"
    return {"dir": s, "spec": f"{s}/{shard}.json", "script": f"{s}/{shard}.start.sh", "pid": f"{s}/{shard}.pid",
            "pgid": f"{s}/{shard}.pgid", "exit": f"{s}/{shard}.exit", "log": f"{workdir}/logs/{shard}.log",
            "tape": f"{s}/events-{shard}.jsonl", "logs": f"{workdir}/logs", "marker": f"{s}/{shard}.marker"}


def start_marker(run_id: str, shard: str, ticket: str) -> str:
    """The marker of one start: ``yeto-ssh-sampler:<run>:<shard>:<ticket>``. The sampler
    has it in its environment (:data:`MARK_ENV`), the session leader as ``$0``, and the
    node keeps it next to the PID file; a stop acts only on processes that carry it."""
    marker = f"{MARKER}:{run_id}:{shard}:{ticket}"
    if any(c.isspace() or c in "'\"\\$`" for c in marker):
        raise ValueError(f"start marker {marker!r}: run, shard and ticket must be plain words")
    return marker


def container_name(shard: str) -> str:
    return f"yeto-sampler-{shard}"


def pool_profile(node: Node, conf: SshPoolConf) -> NodeProfile:
    """The node's profile as this pool starts it: the pool's container engine
    (``runtime: container``) or a host process (``runtime: venv``)."""
    engine = conf.container.engine if conf.runtime == "container" and conf.container else "process"
    return NodeProfile.from_node(profile_of(node), container=engine)


# require_idle: the command that lists the processes on the cards, per unified family
# (ascend reads the process table of ``npu-smi info``, the card tool's own output)
def idle_required(node: Node, conf: SshPoolConf) -> bool:
    """``preflight.require_idle`` for this node: as set, else on for an ascend node only
    (the npu-smi check of before; the nvidia-smi / amd-smi checks are opt-in)."""
    return profile_of(node).family == "ascend" if conf.require_idle is None else conf.require_idle


def idle_command(node: Node) -> str | None:
    family = profile_of(node).family
    if family in ("cuda", "ppu"):
        smi = profile_of(node).hw.smi
        return (f"{smi} --query-gpu=index,pci.bus_id --format=csv,noheader; echo YETO_SSH_APPS; "
                f"{smi} --query-compute-apps=gpu_bus_id,pid --format=csv,noheader")
    if family == "rocm":
        return "amd-smi process --json"
    return None


def preflight_script(node: Node, conf: SshPoolConf) -> str:
    """Node shell: card tool, ``uname -m``, the family's host paths, free GiB of the
    workdir's file system, and with ``require_idle`` the processes on the cards.
    Prints sections the launcher checks (:func:`check_preflight`)."""
    profile = profile_of(node)
    commands = node.preflight_commands()
    # npu: npu-smi info only (its process table is the idle check); rocm: amd-smi, else rocm-smi
    commands = commands if profile.family == "rocm" else commands[:1]
    tools = " || ".join(shlex.join(c) for c in commands) or "true"
    paths = profile.host_paths()
    probe = (f"for f in {' '.join(paths)}; do [ -e $f ] && echo YETO_SSH_HOSTPATH $f; done; " if paths else "")
    idle = idle_command(node) if idle_required(node, conf) else None
    idle_part = f"; echo YETO_SSH_IDLE_BEGIN; {{ {idle}; }} 2>&1; echo YETO_SSH_IDLE_END rc=$?" if idle else ""
    return (f"set -u; mkdir -p {shlex.quote(conf.workdir)}/sampler {shlex.quote(conf.workdir)}/logs || exit 3; "
            f"echo YETO_SSH_ARCH $(uname -m); {probe}"
            f"echo YETO_SSH_FREE_GIB $(df -Pk {shlex.quote(conf.workdir)} | awk 'NR==2{{printf \"%d\", $4/1048576}}'); "
            f"echo YETO_SSH_CARDS_BEGIN; {tools} 2>&1; echo YETO_SSH_CARDS_END rc=$?{idle_part}")


def check_preflight(node: Node, conf: SshPoolConf, text: str) -> dict[str, Any]:
    """The preflight output against the inventory and the provider's limits;
    raises :class:`PreflightError` with every problem at once."""
    problems: list[str] = []
    facts: dict[str, Any] = {}
    cards_text = text.partition("YETO_SSH_CARDS_BEGIN")[2].partition("YETO_SSH_CARDS_END")[0]
    idle_text = text.partition("YETO_SSH_IDLE_BEGIN")[2].partition("YETO_SSH_IDLE_END")[0]
    for line in text.splitlines():
        if line.startswith("YETO_SSH_HOSTPATH "):
            facts.setdefault("host_paths", []).append(line.split(" ", 1)[1].strip())
        elif line.startswith("YETO_SSH_ARCH "):
            facts["arch"] = line.split(" ", 1)[1].strip()
        elif line.startswith("YETO_SSH_FREE_GIB "):
            try:
                facts["free_gib"] = int(line.split(" ", 1)[1].strip())
            except ValueError:
                facts["free_gib"] = None
    try:
        node.check_cpu_arch(facts.get("arch", ""))
    except ValueError as error:
        problems.append(str(error))
    free = facts.get("free_gib")
    if free is None or free < conf.min_free_gib:
        problems.append(f"{conf.workdir}: {free} GiB free, the provider asks {conf.min_free_gib:g} GiB")
    try:
        cards = node.check_cards(cards_text)
        facts["cards"] = len(cards)
    except Exception as error:  # noqa: BLE001 - the card tool's own refusal, as a problem
        problems.append(f"card check: {error}")
        cards = []
    want = conf.min_cards if conf.min_cards is not None else len(conf.devices or ()) or None
    if want is not None and cards and len(cards) < want:
        problems.append(f"{len(cards)} cards, the provider asks {want}")
    missing = [p for p in pool_profile(node, conf).required_host_paths()
               if not node.docker_args and p not in facts.get("host_paths", ())]
    if missing:
        problems.append(f"{', '.join(missing)} missing on the host: a {profile_of(node).family} container "
                        "cannot reach the cards")
    if idle_required(node, conf):
        family = profile_of(node).family
        if family == "ascend":
            busy, tool = busy_npus(cards_text), "npu-smi"
        elif family == "rocm":
            busy, tool = busy_amd(idle_text), "amd-smi process"
        else:
            busy, tool = busy_cuda(idle_text), f"{profile_of(node).hw.smi} compute apps"
        if busy is None:  # the tool did not answer: not checked (the card check above still ran)
            facts["busy"] = None
            facts["idle"] = f"not checked: {tool} gave no process list"
        else:
            mine = set(conf.devices) if conf.devices is not None else None
            busy = [b for b in busy if mine is None or b in mine]
            facts["busy"] = busy
            if busy:
                problems.append(f"cards {busy} run other processes ({tool}); preflight.require_idle is on")
    if problems:
        raise PreflightError(f"node {node.name}: preflight refused: " + "; ".join(problems))
    return facts


def busy_npus(text: str) -> list[int]:
    """NPU ids with a process row in ``npu-smi info`` (the process table: ``| <npu> <chip> | <pid> | ...``)."""
    busy: set[int] = set()
    in_procs = False
    for line in text.splitlines():
        if "Process id" in line:
            in_procs = True
            continue
        if not in_procs or "No running processes" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[1].isdigit():
            head = cells[0].split()
            if head and head[0].isdigit():
                busy.add(int(head[0]))
    return sorted(busy)


def busy_cuda(text: str) -> list[int] | None:
    """Card indexes with a compute process in the output of :func:`idle_command` for
    ``nvidia-smi`` / ``ppu-smi`` (``index, bus id`` rows, ``YETO_SSH_APPS``, then ``bus id,
    pid`` rows); None when the tool listed no card (not installed, no driver)."""
    head, sep, apps = text.partition("YETO_SSH_APPS")
    index_of: dict[str, int] = {}
    for line in head.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 2 and cells[0].isdigit() and ":" in cells[1]:
            index_of[cells[1].lower()] = int(cells[0])
    if not sep or not index_of:
        return None
    busy = set()
    for line in apps.splitlines():
        if not line.strip():
            continue
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != 2 or not cells[1].isdigit():
            return None  # the compute-apps query failed: not "idle"
        if cells[0].lower() in index_of:
            busy.add(index_of[cells[0].lower()])
    return sorted(busy)


def busy_amd(text: str) -> list[int] | None:
    """Card indexes with a process in ``amd-smi process --json`` (a list of ``{"gpu": i,
    "process_list": [...]}``; an entry is ``{"process_info": {..."pid"...}}`` up to ROCm
    6.3 and a flat ``{"pid": ...}`` later; "No running processes detected" is idle).
    None when the output is not that JSON (amd-smi missing or failing)."""
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    try:  # raw_decode: a stderr line after the JSON does not spoil it
        doc = json.JSONDecoder().raw_decode(text[min(starts):])[0] if starts else None
    except ValueError:
        return None
    if isinstance(doc, dict):  # a wrapper object: its first list value
        doc = next((v for v in doc.values() if isinstance(v, list)), None)
    if not isinstance(doc, list):
        return None
    busy = set()
    for entry in doc:
        if not isinstance(entry, dict) or not str(entry.get("gpu", "")).isdigit():
            continue
        for proc in entry.get("process_list") or []:
            info = proc.get("process_info", proc) if isinstance(proc, dict) else None
            pid = info.get("pid") if isinstance(info, dict) else None
            if pid not in (None, "", "N/A"):
                busy.add(int(entry["gpu"]))
    return sorted(busy)


def _alive(p: Mapping[str, str]) -> str:
    """Shell test: the PID in the PID file lives, is an engine sampler (``ps -o args=``)
    and carries this start's marker (the marker file) in its environment, so a reused
    PID (another run's sampler, or anything else) is never taken for ours. Sets
    ``$p`` (PID) and ``$m`` (marker)."""
    return (f"{{ p=$(cat {p['pid']} 2>/dev/null); m=$(cat {p['marker']} 2>/dev/null); "
            "[ -n \"$p\" ] && [ -n \"$m\" ] && kill -0 \"$p\" 2>/dev/null && "
            "ps -o args= -p \"$p\" 2>/dev/null | grep -q yeto.rl.engine.engine_sampler && "
            f"tr '\\0' '\\n' < /proc/$p/environ 2>/dev/null | grep -qxF \"{MARK_ENV}=$m\"; }}")


def _group_ok(p: Mapping[str, str]) -> str:
    """Shell test: the recorded session leader still has this start's marker as its
    ``$0`` (``ps -o args=``, fixed string). Sets ``$g`` (PGID)."""
    return (f"{{ g=$(cat {p['pgid']} 2>/dev/null); m=$(cat {p['marker']} 2>/dev/null); "
            "[ -n \"$g\" ] && [ -n \"$m\" ] && ps -o args= -p \"$g\" 2>/dev/null | grep -qF -- \"$m\"; }")


def start_script(conf: SshPoolConf, *, yeto_dir: str, spec_path: str, pid_path: str,
                 env: Mapping[str, str], env_file: str | None, exec_line: str | None = None) -> str:
    """The venv start script: env scripts, environment, PID record, ``exec`` python.
    ``exec_line``: the node runtime's start line instead (it records the sampler's PID itself)."""
    assert conf.venv is not None
    v = conf.venv
    lines = ["#!/bin/bash", "set -e"]
    lines += [f"source {shlex.quote(p)}" for p in v.source]  # set_env.sh scripts read unset variables
    lines.append("set -u")
    if env_file:
        lines += ["set -a", f". {shlex.quote(env_file)}", "set +a"]
    lines += [f"export {k}={shlex.quote(val)}" for k, val in sorted(env.items())]
    path = ":".join([yeto_dir, *v.pythonpath])
    lines.append(f'export PYTHONPATH={shlex.quote(path)}"${{PYTHONPATH:+:$PYTHONPATH}}"')
    if v.ld_preload:
        lines.append(f"export LD_PRELOAD={shlex.quote(v.ld_preload)}")
    if env_file:
        lines.append(_STORE_URI)
    if exec_line is not None:
        lines += [f"cd {shlex.quote(conf.workdir)}", exec_line]
    else:
        lines += [f"cd {shlex.quote(conf.workdir)}", f"echo $$ > {shlex.quote(pid_path)}",
                  f"exec {shlex.quote(v.python)} -m yeto.rl.engine.engine_sampler --spec {shlex.quote(spec_path)}"]
    return "\n".join(lines) + "\n"


_STORE_URI = STORE_URI_SHELL  # the bucket and endpoint are only known on the node (its env file)


def venv_launch_command(p: Mapping[str, str], marker: str, wait_s: int = 10) -> str:
    """Detach the start script in its own session (``$0``: the start's marker); the
    session's PGID and the marker are recorded, the script records the sampler's own
    PID. The log starts empty: a line of an earlier run is never read as this one's."""
    inner = f"bash {shlex.quote(p['script'])} >> {shlex.quote(p['log'])} 2>&1; echo $? > {shlex.quote(p['exit'])}"
    return (f"rm -f {p['exit']} {p['pid']}; mkdir -p {p['logs']}; : > {p['log']}; "
            f"printf '%s\\n' {shlex.quote(marker)} > {p['marker']}; "
            f"{{ setsid bash -c {shlex.quote(inner)} {shlex.quote(marker)} </dev/null >/dev/null 2>&1 & "
            f"echo $! > {p['pgid']}; }}; "
            f"for i in {' '.join(str(i) for i in range(1, int(wait_s) + 1))}; do {_alive(p)} && break; sleep 1; done; "
            f"if {_alive(p)}; then echo YETO_SSH_STARTED pid=$(cat {p['pid']}) pgid=$(cat {p['pgid']}); "
            f"else echo YETO_SSH_START_FAILED; tail -n 40 {p['log']} 2>/dev/null; exit 1; fi")


# the sampler's line once its LEAVE is durable after a SIGTERM (engine_sampler_entry.hold_after_reclaim):
# it then only waits for the platform to stop it, so the stop need not wait out the grace
LEFT_LINE = "reclaimed: waiting up to"


def venv_stop_command(p: Mapping[str, str], grace_s: int) -> str:
    """SIGTERM to the recorded sampler PID when it is this start's (:func:`_alive`);
    wait until it exits or has left (a :data:`LEFT_LINE` *after* the TERM: the count
    is taken before it), at most ``grace_s``; then KILL that PID, and TERM / KILL the
    rest of the start's process group only when its leader still carries the marker."""
    left = f"grep -c '{LEFT_LINE}' {p['log']} 2>/dev/null"
    return (f"if {_alive(p)}; then n0=$({left}); n0=${{n0:-0}}; echo YETO_SSH_TERM pid=$p; kill -TERM \"$p\"; "
            f"for i in $(seq {int(grace_s)}); do kill -0 \"$p\" 2>/dev/null || break; "
            f"n=$({left}); [ \"${{n:-0}}\" -gt \"$n0\" ] && {{ echo YETO_SSH_LEFT after=$i; break; }}; "
            "sleep 1; done; "
            f"{_alive(p)} && {{ echo YETO_SSH_KILL pid=$p; kill -KILL \"$p\"; }}; fi; "
            f"if {_group_ok(p)} && kill -0 -- -\"$g\" 2>/dev/null; then echo YETO_SSH_GROUP_TERM pgid=$g; "
            "kill -TERM -- -\"$g\" 2>/dev/null; for i in $(seq 30); do kill -0 -- -\"$g\" 2>/dev/null || break; "
            f"sleep 1; done; {_group_ok(p)} && kill -KILL -- -\"$g\" 2>/dev/null; fi; "
            f"rm -f {p['pid']} {p['pgid']}; echo YETO_SSH_EXIT $(cat {p['exit']} 2>/dev/null || echo -); "
            "echo stopped")


def container_stop_command(conf: "SshPoolConf", p: Mapping[str, str], name: str, grace_s: int) -> str:
    """SIGTERM to the container's PID 1 (the sampler); wait until it exits or has left
    (a :data:`LEFT_LINE` after the TERM: counted before it), at most ``grace_s``; then
    ``stop`` (TERM, KILL after 10 s), keep its log, remove it."""
    ce = _ce(conf)
    running = f"[ \"$({ce} inspect -f '{{{{.State.Status}}}}' {name} 2>/dev/null)\" = running ]"
    left = f"{ce} logs {name} 2>&1 | grep -c '{LEFT_LINE}'"
    return (f"mkdir -p {p['logs']}; if {ce} inspect --type container {name} >/dev/null 2>&1; then "
            f"if {running}; then n0=$({left}); n0=${{n0:-0}}; echo YETO_SSH_TERM container={name}; "
            f"{ce} kill --signal TERM {name} >/dev/null 2>&1; "
            f"for i in $(seq {int(grace_s)}); do {running} || break; "
            f"n=$({left}); [ \"${{n:-0}}\" -gt \"$n0\" ] && {{ echo YETO_SSH_LEFT after=$i; break; }}; "
            "sleep 1; done; fi; "
            f"{ce} stop -t 10 {name} >/dev/null 2>&1; "
            f"echo YETO_SSH_EXIT $({ce} inspect -f '{{{{.State.ExitCode}}}}' {name} 2>/dev/null); "
            f"{ce} logs {name} > {p['logs']}/{name}.last.log 2>&1; {ce} rm -f {name} >/dev/null 2>&1; fi; "
            "echo stopped")


def venv_state_command(p: Mapping[str, str]) -> str:
    return (f"if {_alive(p)}; then echo 'running -'; elif [ -f {p['exit']} ]; then "
            f"echo \"exited $(cat {p['exit']})\"; elif {_group_ok(p)} && kill -0 -- -\"$g\" 2>/dev/null; then echo 'running -'; "
            "else echo 'absent -'; fi")


def _ce(conf: SshPoolConf) -> str:
    assert conf.container is not None
    return ("sudo -n " if conf.container.sudo else "") + conf.container.engine


def container_state_command(conf: SshPoolConf, name: str) -> str:
    inspect = f"{_ce(conf)} inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {name}"
    return (f"{inspect} 2>/dev/null || case \"$({inspect} 2>&1 >/dev/null)\" in "
            "(*'o such'*|*'unable to find'*|*'no container'*) echo 'absent -';; (*) echo 'unknown -';; esac")


def container_pull_command(conf: SshPoolConf, image: str) -> str:
    c = conf.container
    assert c is not None
    have = f"{_ce(conf)} inspect --type image {shlex.quote(image)} >/dev/null 2>&1"
    pull = f"{_ce(conf)} pull {shlex.quote(c.pull_ref or image)}"
    if c.pull == "never":
        return f"{have} || {{ echo YETO_SSH_NO_IMAGE {shlex.quote(image)}; exit 4; }}"
    if c.pull == "always":
        return f"{pull} >/dev/null && {have}"
    return f"{have} || {{ {pull} >/dev/null && {have}; }} || {{ echo YETO_SSH_NO_IMAGE {shlex.quote(image)}; exit 4; }}"


def container_run_argv(conf: SshPoolConf, node: Node, *, name: str, image: str, yeto_dir: str, spec_path: str,
                       env: Mapping[str, str], env_file: str | None, cards: Sequence[int],
                       host_paths: Sequence[str] | None = None, command: str | None = None) -> list[str]:
    """``<engine> run -d``: the node's device flags (the inventory's ``docker_args``,
    else the defaults of its family for ``cards``: :meth:`NodeProfile.devices`),
    ``security_opt``, the workdir and mounts at the same paths as on the host, then
    ``exec`` of the engine sampler as PID 1."""
    c = conf.container
    assert c is not None
    argv = [*(["sudo", "-n"] if c.sudo else []), c.engine, "run", "-d", "--name", name,
            "--network", "host", "--ipc", "host", "--shm-size", c.shm_size]
    for opt in c.security_opt:
        argv += ["--security-opt", opt]
    # what the preflight found on the host is passed (a missing -v source fails the run); the pool's
    # container.shm_size above is the one shm size (the rocm defaults carry the NC route's 64g)
    if node.docker_args:  # the site's flags, verbatim (as before)
        argv += list(node.docker_args)
    else:
        argv += _without_shm(pool_profile(node, conf).devices(cards, engine=c.engine, host_paths=host_paths).args)
    mounts = {conf.workdir: conf.workdir, **dict(node.mounts), **dict(c.mounts)}
    if conf.store and not conf.store.startswith(conf.workdir + "/"):
        mounts.setdefault(conf.store, conf.store)
    for host, inside in sorted(mounts.items()):
        argv += ["-v", f"{host}:{inside}"]
    if env_file:
        argv += ["--env-file", env_file]
    for key, value in sorted(env.items()):
        argv += ["-e", f"{key}={value}"]
    argv += list(c.args)
    # the yeto tree goes in front of the image's own PYTHONPATH (the CANN ``acl`` module of an
    # Ascend image is found through it; ``-e PYTHONPATH=`` would drop it, A07 pit 3)
    exec_line = command or f"exec {shlex.quote(c.python)} -m yeto.rl.engine.engine_sampler --spec {shlex.quote(spec_path)}"
    body = f'export PYTHONPATH={shlex.quote(yeto_dir)}"${{PYTHONPATH:+:$PYTHONPATH}}"; ' + \
        ((_STORE_URI + "; ") if env_file else "") + exec_line
    return [*argv, "--entrypoint", "bash", image, "-c", body]


def device_env(node: Node, conf: SshPoolConf) -> dict[str, str]:
    """The sampler's card variables: the visible-card variable only for an explicit
    ``devices`` list (numbered as the container sees the cards; a venv's own env
    scripts are not overridden otherwise), ``YETO_DEVICE_FAMILY`` on a cuda / rocm / ppu
    node. A node with ``docker_args`` gets only the ascend variable it always got."""
    profile = pool_profile(node, conf)
    env: dict[str, str] = {}
    if conf.devices is not None:
        if node.docker_args:  # as before: the site's flags pass the cards, only ascend was numbered
            if profile.family == "ascend":
                env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(i) for i in conf.devices)
        else:
            env.update(profile.devices(list(conf.devices)).env)
    if profile.family != "ascend" and not node.docker_args:
        env.update(profile.family_env())
    return env


def _without_shm(args: Sequence[str]) -> list[str]:
    """``args`` without ``--shm-size X`` / ``--shm-size=X`` (the pool sets its own)."""
    out: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg == "--shm-size":
            skip = True
        elif not arg.startswith("--shm-size="):
            out.append(arg)
    return out


# -- the provider ---------------------------------------------------------------


class SshNodeProvider:
    """Slots are the configured nodes; tickets live in a local record. Never orders,
    never powers anything off: ``release`` only marks the ticket ended (the ops stop
    the sampler first).

    Spare nodes: the pool's slots may be more than its target (every node the inventory
    offers). A member whose node was lost (the pool judged it unreachable, or its stop
    did not reach the node) is released with :meth:`mark_lost`: its slot is held for
    ``lost_node_hold_s``, and while any other slot's node is free :meth:`account_free`
    reports the held one busy, so the pool's replacement (fenced first, continuing the
    lost shard's prompts: ``IslandPool._retire``) goes to a spare node, not back to the
    lost one. Without a free spare the held node is offered as before. A stop the lost
    node never got is owed (:meth:`owed_stops`): the next start there stops that sampler
    first, so a node back from a partition never runs two samplers.

    ``lost_node_hold_s: 0`` turns the hold off: a lost node's slot is offered like any
    free slot at once (the owed stop is still recorded and still runs first). A drain
    whose stop hit one ssh failure (exit 255 after ``ssh_transport``'s own retries of a
    dropped connection, or a timeout) counts as a loss too: that costs at most a move
    to a spare, never a missing sampler (the hold yields when no spare is free).

    Where the record lives: on the local control plane, this provider's state file on
    the control host (``state_path``). On the store control plane (:meth:`attach_store`,
    from the pool's ``store_state.PoolControl``) it is in the run store, one record per
    holder of the pool's store lock: ``fleet/pools/<pool>/provider/<token>.json``. A holder
    writes only its own token's record, each save after ``StoreLock.guard`` (a fenced
    holder raises ``LockLost`` and writes nothing); readers take the record of the highest
    token. A controller that takes the pool over (on another host) reads the newest earlier
    record and writes its own at once: the tickets, the holds and the owed stops carry over,
    and the local file is imported only when the store has no record at all (from then on
    it is only mirrored). A write of the fenced holder that lands late (its guard passed,
    then it stalled past its lease while the next holder claimed the pool and read)
    replaces nothing, and is not lost either: every save of the next holder merges the
    newest earlier record's lost-node holds and owed stops into its own
    (:meth:`_merge_predecessor`), so a sampler the fenced holder left on a lost node is
    still stopped before the next start there."""

    kind = KIND
    caps = provider_caps(KIND)  # yeto/cloud/capabilities.json

    def __init__(self, run_id: str, conf: SshPoolConf, *, slots: Sequence[int], state_path: Path,
                 clock: Callable[[], float] = time.time):
        self.run_id, self.conf, self.slots, self.state_path = run_id, conf, list(slots), state_path
        self.clock = clock
        self._lock = threading.Lock()
        # the store control plane (attach_store): (objects, key, lock) and the record in memory
        self._store: tuple[Any, str, Any] | None = None
        self._store_rows: dict[str, dict[str, Any]] = {}
        # tickets whose owed stop went through (stop_done): a predecessor's record that still
        # owes it never brings it back (kept in the record: a successor reads them too)
        self._stops_done: set[str] = set()
        self.imported_from: str | None = None
        bad = [s for s in self.slots if not 0 <= s < len(conf.nodes)]
        if bad:
            raise ValueError(f"ssh provider: slots {bad} have no node (nodes: {[n.name for n in conf.nodes]})")

    def attach_store(self, objects: Any, pool: str, lock: Any) -> dict[str, Any]:
        """Keep the record in the run store under pool ``pool``'s store ``lock`` (class doc):
        the newest record of this or an earlier lock holder, else the local file imported once; the
        record is written under this holder's token at once (a later attach elsewhere finds
        it, not that host's file). Returns where the record came from (``source``: ``store``,
        ``local`` or ``new``, and ``key`` / ``path``)."""
        from .pool import store_state  # noqa: PLC0415

        found = store_state.read_provider_record(objects, pool, upto=lock.token)  # unreadable: raises
        if found is not None:
            rows, origin = found[1]["tickets"], {"source": "store", "key": found[0]}
        elif self.state_path.exists():
            rows, origin = self._local_rows(), {"source": "local", "path": str(self.state_path)}
        else:
            rows, origin = {}, {"source": "new"}
        with self._lock:
            self._store = (objects, store_state.provider_record_key(pool, lock.token), lock)
            self._store_pool = pool
            self._store_rows = {t: dict(r) for t, r in rows.items()}
            self._stops_done = {str(t) for t in (found[1].get("stops_done") or ())} if found else set()
            self.imported_from = origin.get("path") or (found[1].get("imported_from") if found else None)
            self._save(self._rows())
        self._prune(objects, pool, lock.token)
        return origin

    @staticmethod
    def _prune(objects: Any, pool: str, token: int) -> None:
        """Best effort: drop the records of holders before the previous one, and the first
        version's single record (this holder's record is in the store now)."""
        from .pool import store_state  # noqa: PLC0415

        try:
            older = [t for t in store_state.provider_tokens(objects, pool) if t < token]
            for t in older[:-1]:
                objects.delete(store_state.provider_record_key(pool, t))
            if objects.read(store_state.provider_key(pool)) is not None:
                objects.delete(store_state.provider_key(pool))
        except Exception:  # noqa: BLE001 - never read past the newest record: pruned at the next attach
            pass

    def _local_rows(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def _rows(self) -> dict[str, dict[str, Any]]:
        if self._store is not None:  # the lock holder's record (a copy: a save that fails changes nothing)
            return {t: dict(r) for t, r in self._store_rows.items()}
        return self._local_rows()

    def _save(self, rows: dict[str, dict[str, Any]]) -> None:
        if self._store is not None:
            objects, key, lock = self._store
            lock.guard()  # LockLost once another controller took the pool; a store blip inside the lease passes
            rows = self._merge_predecessor(objects, lock.token, rows)
            doc = {"tickets": rows, "control": {"token": lock.token, "holder": lock.holder, "saved_at": self.clock()},
                   **({"imported_from": self.imported_from} if self.imported_from else {}),
                   **({"stops_done": sorted(self._stops_done)} if self._stops_done else {})}
            objects.write(key, json.dumps(doc, indent=1, sort_keys=True).encode())
            self._store_rows = {t: dict(r) for t, r in rows.items()}
            try:  # the local mirror (status readers on this host); never read back on this plane
                self._write_local(rows)
            except OSError:
                pass
            return
        self._write_local(rows)

    def _merge_predecessor(self, objects: Any, token: int, rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """``rows`` with the lost-node holds and owed stops of the newest record of an
        earlier lock holder merged in. A fenced holder's write can land after this holder
        read the store (class doc): its hold, and above all its owed stop (a sampler left
        on a node, billing, that nobody would stop), are not lost. Holds: the later
        ``held_until``; owed stops: the union, less those this holder saw go through
        (``stops_done``, its predecessors' too). A ticket only the predecessor knows comes
        in as a released row with its hold / owed stop. Its other fields (state, the
        tickets it ordered) are this holder's. A predecessor record that does not read
        is not merged this time (the next save reads it again)."""
        from .pool import store_state  # noqa: PLC0415

        try:
            found = store_state.read_provider_record(objects, self._store_pool, upto=int(token) - 1)
        except Exception:  # noqa: BLE001 - unreadable now: merged at a later save
            return rows
        if found is None:
            return rows
        done = self._stops_done | {str(t) for t in (found[1].get("stops_done") or ())}
        out = {t: dict(r) for t, r in rows.items()}
        for ticket, theirs in (found[1].get("tickets") or {}).items():
            held = float(theirs.get("held_until") or 0.0)
            owed = bool(theirs.get("stop_owed")) and ticket not in done and theirs.get("shard")
            if not held and not owed:
                continue
            mine = out.get(ticket)
            if mine is None:
                mine = {k: theirs[k] for k in ("slot", "node", "shard", "ordered_at", "lost_at") if k in theirs}
                mine["state"] = "released"
            if held > float(mine.get("held_until") or 0.0):
                mine["held_until"] = held
                mine.setdefault("lost_at", theirs.get("lost_at"))
            if owed and not mine.get("stop_owed"):
                mine["stop_owed"] = True
                mine.setdefault("shard", theirs.get("shard"))
            out[ticket] = mine
        return out

    def _write_local(self, rows: dict[str, dict[str, Any]]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=1, sort_keys=True))
        tmp.replace(self.state_path)

    def node_of_slot(self, slot: int) -> Node:
        return self.conf.nodes[int(slot)]

    def node_of_ticket(self, ticket: str) -> Node:
        row = self._rows().get(ticket)
        if row is None:
            raise KeyError(f"ticket {ticket} is not a ticket of this pool")
        return self.node_of_slot(int(row["slot"]))

    def order(self, account: int) -> str:
        with self._lock:
            rows = self._rows()
            n = 1 + sum(1 for r in rows.values() if int(r["slot"]) == int(account))
            ticket = f"{self.run_id}-{self.node_of_slot(account).name}-{n}"
            rows[ticket] = {"slot": int(account), "node": self.node_of_slot(account).name, "state": "active",
                            "ordered_at": self.clock()}
            self._save(rows)
            return ticket

    def status(self, account: int, ticket: str) -> LeaseStatus:  # noqa: ARG002
        row = self._rows().get(ticket)
        if row is None:
            return LeaseStatus("ended", end_reason="lost")
        if row["state"] == "released":
            return LeaseStatus("ended", end_reason="released")
        return LeaseStatus("active")

    def release(self, account: int, ticket: str) -> dict[str, Any]:  # noqa: ARG002
        with self._lock:
            rows = self._rows()
            if ticket in rows:
                rows[ticket] = dict(rows[ticket], state="released", released_at=self.clock())
                self._save(rows)
        return {"ticket": ticket, "state": "released"}

    def released_by_us(self, ticket: str) -> bool:
        return self._rows().get(ticket, {}).get("state") == "released"

    def node(self, account: int, ticket: str) -> Node:  # noqa: ARG002
        return self.node_of_ticket(ticket)

    def balance(self) -> int | None:
        return None

    def account_free(self, account: int) -> bool:
        rows, now = self._rows(), self.clock()
        if self._busy(rows, account):
            return False
        if self._held(rows, account, now) is None:
            return True
        # a lost node: offered again only when no other slot's node is free (no spare left)
        return not any(not self._busy(rows, s) and self._held(rows, s, now) is None
                       for s in self.slots if int(s) != int(account))

    @staticmethod
    def _busy(rows: Mapping[str, Mapping[str, Any]], slot: int) -> bool:
        return any(int(r["slot"]) == int(slot) and r["state"] == "active" for r in rows.values())

    @staticmethod
    def _held(rows: Mapping[str, Mapping[str, Any]], slot: int, now: float) -> float | None:
        """Until when ``slot``'s node is held as lost (None: not held)."""
        until = max((float(r.get("held_until") or 0.0) for r in rows.values() if int(r["slot"]) == int(slot)),
                    default=0.0)
        return until if until > now else None

    def note_shard(self, ticket: str, shard: str) -> None:
        """The shard ``ticket``'s sampler runs (an owed stop names it)."""
        with self._lock:
            rows = self._rows()
            if ticket in rows and rows[ticket].get("shard") != shard:
                rows[ticket] = dict(rows[ticket], shard=shard)
                self._save(rows)

    def mark_lost(self, ticket: str, *, stopped: bool) -> None:
        """``ticket``'s node was lost under its sampler: hold its slot (class doc); a stop
        that did not reach the node is owed to the next start there."""
        with self._lock:
            rows = self._rows()
            if ticket not in rows:
                return
            row = dict(rows[ticket], lost_at=self.clock(), held_until=self.clock() + self.conf.lost_node_hold_s)
            if not stopped:
                row["stop_owed"] = True
            rows[ticket] = row
            self._save(rows)

    def owed_stops(self, slot: int) -> list[tuple[str, str]]:
        """``(ticket, shard)`` of samplers on ``slot``'s node whose stop never reached it."""
        return sorted((t, str(r["shard"])) for t, r in self._rows().items()
                      if int(r["slot"]) == int(slot) and r.get("stop_owed") and r.get("shard"))

    def stop_done(self, ticket: str) -> None:
        with self._lock:
            rows = self._rows()
            if ticket in rows and rows[ticket].get("stop_owed"):
                rows[ticket] = {k: v for k, v in rows[ticket].items() if k != "stop_owed"}
                if self._store is not None:
                    self._stops_done.add(ticket)
                self._save(rows)

    def run_tickets(self) -> list[tuple[int, str]]:
        return [(int(r["slot"]), t) for t, r in self._rows().items() if r["state"] == "active"]

    def hourly_cost(self, ticket: str | None = None) -> float | None:  # noqa: ARG002
        return self.conf.price_usd_h

    # -- group order (yeto.launch.pool.group): n free inventory nodes of one card and one fabric
    @property
    def fabrics(self) -> tuple[str, ...]:
        have = {self.node_of_slot(s).fabric for s in self.slots}
        return tuple(sorted(have | ({"rdma"} if have & {"ib", "roce"} else set())))

    def order_group(self, n: int, shape: Any, fabric: str = "tcp") -> list[tuple[int, str]]:
        """Pick ``n`` free nodes with the shape's card and card count that share one fabric
        (``fabric`` itself; ``rdma``: one of ib / roce; ``tcp``: any one fabric), in slot
        order, and order each. A failure releases the nodes ordered so far."""
        def fits(node: Node) -> bool:
            return (str(node.card).lower() == str(getattr(shape, "gpu", node.card)).lower()
                    and int(node.cards_per_node) == int(getattr(shape, "gpus_per_node", node.cards_per_node)))

        by_fabric: dict[str, list[int]] = {}
        for slot in self.slots:
            node = self.node_of_slot(slot)
            if fits(node) and self.account_free(slot) and (
                    fabric in ("tcp", node.fabric) or (fabric == "rdma" and node.fabric in ("ib", "roce"))):
                by_fabric.setdefault(node.fabric, []).append(slot)
        pick = next((slots for slots in by_fabric.values() if len(slots) >= int(n)), None)
        if pick is None:
            raise RuntimeError(f"ssh inventory: no {n} free {getattr(shape, 'gpus_per_node', '?')} x "
                               f"{getattr(shape, 'gpu', '?')} nodes on one {fabric} fabric "
                               f"(free by fabric: {by_fabric})")
        from .pool.group import release_placed

        placed: list[tuple[int, str]] = []
        try:
            for slot in pick[:int(n)]:
                placed.append((slot, self.order(slot)))
        except Exception as error:
            raise release_placed(self, placed, error) from error
        return placed


# -- the pool ops ---------------------------------------------------------------


class SshSamplerPoolOps:
    """The pool's ``PoolOps`` for samplers on the operator's own SSH nodes."""

    start_in_background = True  # an engine cold start on an NPU takes minutes

    def __init__(self, cfg: dict[str, Any], *, conf: SshPoolConf | None = None,
                 run: Callable[..., Any] = subprocess.run, state_dir: Path | None = None,
                 code_ref: str = "HEAD", clock: Callable[[], float] = time.time):
        from .nc import cli  # noqa: PLC0415

        self.cfg = cfg
        self.run_id = cli.run_id(cfg)
        self.conf = conf if conf is not None else parse_config(cfg)
        self._run = run
        self.code_ref = code_ref
        if state_dir is None:
            from .nc.tunnel import state_dir as nc_state_dir  # noqa: PLC0415

            state_dir = nc_state_dir()
        self.state_dir = Path(state_dir)
        self.provider = SshNodeProvider(self.run_id, self.conf, slots=cli.slot_list(cfg),
                                        state_path=self.state_dir / f"ssh-pool-{self.run_id}.json", clock=clock)
        self._shard_of: dict[str, str] = {}
        self._node_route: Any = None

    def attach_control(self, control: Any) -> dict[str, Any]:
        """The store control plane (``store_state.PoolControl``): the provider's tickets, holds
        and owed stops move into the run store next to the pool's state, under its lock
        (:meth:`SshNodeProvider.attach_store`; returns where the record came from)."""
        return self.provider.attach_store(control.state.objects, control.state.pool, control.lock)

    def node_route_for(self, node: Node) -> Any:
        """The node runtime's assignments (``yeto.node.route``): in the run's object store, or
        in the node-local store (``provider.store``, written through the node's shell). None:
        ``node_runtime: false``, or the syncer data path (no store): the node then starts the
        sampler itself, as before."""
        if not self.conf.node_runtime:
            return None
        if self.conf.store:
            from yeto.node.route import NodeRoute, ShellObjects  # noqa: PLC0415

            objects = ShellObjects(lambda cmd, text: self.ssh(node, cmd, input=text, timeout=120), self.conf.store)
            return NodeRoute(self.cfg, kind=KIND, objects=lambda: objects, store=lambda: str(self.conf.store),
                             code_ref=self.code_ref, run=self._run, publish_code=False)
        if not self._uses_r2():
            return None
        if self._node_route is None:
            from yeto.node.route import NodeRoute  # noqa: PLC0415
            from yeto.node.store import run_store_uri  # noqa: PLC0415

            self._node_route = NodeRoute(
                self.cfg, kind=KIND, objects=lambda: self._fence_store().objects, code_ref=self.code_ref,
                store=lambda: run_store_uri(self.cfg["r2_env_file"], self.cfg["run_prefix"],
                                            bucket_env=self.cfg.get("bucket_env", "R2_BUCKET")), run=self._run)
        return self._node_route

    def node_profile(self, node: Node) -> Any:
        """The node's profile for the runtime it starts (a process there: inside the container
        of a container pool, on the host of a venv pool; the workdir on the same path in both)."""
        from yeto.launch.node_inventory import NodePaths, NodeProfile  # noqa: PLC0415

        return NodeProfile.from_node(node, container="process",
                                     paths=NodePaths(work=self.conf.workdir, models=self.conf.workdir,
                                                     cache=f"{self.conf.workdir}/cache"))

    # -- shell on a node
    def ssh(self, node: Node, command: str, *, timeout: float = 600, input: str | bytes | None = None,
            stdin: Any = None) -> Any:
        route = [*node_ssh_options(node), node.host]  # its reach: ProxyJump (jump), the relay's ProxyCommand
        argv = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", *ssh_transport.mux_options(route),
                *route, command]
        if input is None and stdin is None:
            argv.insert(1, "-n")
        kw: dict[str, Any] = {"capture_output": True, "timeout": timeout, "check": False}
        if stdin is not None:  # a stream is read once: no retry
            kw = {"stdin": stdin, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "timeout": timeout,
                  "check": False}
            return self._run(argv, **kw)
        kw["text"] = not isinstance(input, bytes)
        kw["input"] = input
        return ssh_transport.run_with_retry(self._run, argv, label=node.name, log=lambda line: print(line, flush=True),
                                            **kw)

    def put(self, node: Node, text: str, remote: str, *, mode: str = "644") -> None:
        out = self.ssh(node, f"umask 077; cat > {shlex.quote(remote)}.tmp && chmod {mode} {shlex.quote(remote)}.tmp "
                             f"&& mv -f {shlex.quote(remote)}.tmp {shlex.quote(remote)}", input=text, timeout=120)
        if out.returncode != 0:
            raise SshSamplerError(f"upload {remote} to {node.name}: {_text(out.stderr)[-300:]}")

    def upload_code(self, node: Node) -> str:
        """``git archive`` (gzip: a slow launcher link) of ``yeto/`` at ``code_ref`` into
        ``<workdir>/yeto/<sha>`` (idempotent: a tree already there is kept)."""
        sha = self._run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short=12", self.code_ref],
                        capture_output=True, text=True, check=True).stdout.strip()
        dest = f"{self.conf.workdir}/yeto/{sha}"
        archive = subprocess.Popen(["git", "-C", str(REPO_ROOT), "archive", "--format=tar.gz", self.code_ref, "yeto"],
                                   stdout=subprocess.PIPE)
        try:
            out = self.ssh(node, f"test -d {dest}/yeto && cat >/dev/null || (rm -rf {dest}.tmp && mkdir -p {dest}.tmp "
                                 f"&& tar xz -C {dest}.tmp && mv {dest}.tmp {dest})", stdin=archive.stdout, timeout=600)
        finally:
            if archive.stdout is not None:
                archive.stdout.close()
            archive.wait()
        if out.returncode != 0:
            raise SshSamplerError(f"code upload to {node.name}: {_text(out.stderr)[-300:]}")
        return dest

    # -- supply
    def order(self, account: int) -> str:
        return self.provider.order(account)

    def poll(self, m) -> LeaseStatus:
        return self.provider.status(m.account, m.ticket)

    def balance(self) -> int | None:
        return None

    def account_free(self, account: int) -> bool:
        return self.provider.account_free(account)

    def run_tickets(self) -> list[tuple[int, str]]:
        return self.provider.run_tickets()

    def released_by_us(self, ticket: str) -> bool:
        return self.provider.released_by_us(ticket)

    def hourly_cost(self, m) -> float | None:  # noqa: ARG002
        return self.conf.price_usd_h

    # -- island
    def cards(self, node: Node) -> list[int]:
        return list(self.conf.devices) if self.conf.devices is not None else list(range(node.cards_per_node))

    def preflight(self, node: Node) -> dict[str, Any]:
        out = self.ssh(node, preflight_script(node, self.conf), timeout=180)
        if out.returncode == 255:
            raise SshSamplerError(f"node {node.name}: ssh failed: {_text(out.stderr)[-300:]}")
        return check_preflight(node, self.conf, _text(out.stdout))

    def spec_for(self, m, shards: list[str]) -> dict[str, Any]:
        from .nc import cli  # noqa: PLC0415

        spec = {"start_on_latest": True, **cli.render_spec(self.cfg, m, shards)}
        p = node_paths(self.conf.workdir, m.shard)
        spec.setdefault("event_tape", p["tape"])
        if self.conf.store:
            spec["store"] = spec["buffer_root"] = self.conf.store
        return spec

    def sampler_env(self, node: Node, m) -> dict[str, str]:
        # SIGTERM is the orderly leave (sampler_resume.install_marker_signals)
        env = {"PYTHONUNBUFFERED": "1", **reclaim_marker_env(m.shard, f"{self.conf.workdir}/sampler")}
        env.update(device_env(node, self.conf))
        if self._uses_r2():
            prefix = str(self.cfg["run_prefix"]).strip("/")
            env.update({"YETO_RL_STORE_IMPL": "yeto.cloud.s3_store:store_from_env",
                        "YETO_RL_STORE_S3_STAGE": f"{self.conf.workdir}/stage", "YETO_RL_STORE_S3_REGION": "auto",
                        "YETO_RL_STORE_S3_CREDENTIALS": "aws-env", "YETO_NC_RUN_PREFIX": prefix,
                        **sampler_route_env(prefix, m.shard, f"{self.conf.workdir}/sampler")})
        env.update(self.conf.env)
        return env

    def _uses_r2(self) -> bool:
        return not self.conf.store and self.cfg.get("data_path", "r2") == "r2"

    def start(self, m, shards: list[str]) -> None:
        node = self.provider.node_of_ticket(m.ticket)
        self._shard_of[m.ticket] = m.shard
        self.provider.note_shard(m.ticket, m.shard)
        p = node_paths(self.conf.workdir, m.shard)
        self._owed_stops(node, m)  # a node back from a loss: its old sampler goes before the preflight looks
        # this shard's earlier sampler first (a restarted pool, an in-place restart): never two samplers of
        # one shard, and the idle check must not see this shard's own cards as another's
        self._stop_on(node, m.shard)
        facts = self.preflight(node)
        _log(f"ssh pool: {node.name} preflight ok {facts}")
        if facts.get("idle"):
            self.event("ssh_pool_idle_unchecked", node=node.name, shard=m.shard, detail=facts["idle"])
        yeto_dir = self.upload_code(node)
        spec = self.spec_for(m, shards)
        route = self.node_route_for(node)
        if route is None:
            self.put(node, json.dumps(spec, indent=1), p["spec"])
        env_file = None
        if self._uses_r2():
            env_file = f"{self.conf.workdir}/sampler/.store.env"
            self.put(node, Path(os.path.expanduser(str(self.cfg["r2_env_file"]))).read_text(), env_file, mode="600")
        env = self.sampler_env(node, m)
        exec_line = None
        if route is not None:  # the node runtime starts the sampler (its records go to the store)
            python = self.conf.venv.python if self.conf.venv else self.conf.container.python  # type: ignore[union-attr]
            exec_line = self.start_node_runtime(route, node, m, shards, spec, python=python, env_file=env_file)
        if self.conf.runtime == "venv":
            marker = start_marker(self.run_id, m.shard, m.ticket)
            env[MARK_ENV] = marker
            self.put(node, start_script(self.conf, yeto_dir=yeto_dir, spec_path=p["spec"], pid_path=p["pid"],
                                        env=env, env_file=env_file, exec_line=exec_line), p["script"], mode="700")
            out = self.ssh(node, venv_launch_command(p, marker, wait_s=10 if exec_line is None else 120),
                           timeout=120 if exec_line is None else 240)
        else:
            image = self.image_for(node)
            name = container_name(m.shard)
            argv = container_run_argv(self.conf, node, name=name, image=str(image), yeto_dir=yeto_dir,
                                      spec_path=p["spec"], env=env, env_file=env_file, cards=self.cards(node),
                                      host_paths=facts.get("host_paths"), command=exec_line)
            out = self.ssh(node, f"{container_pull_command(self.conf, str(image))} && "
                                 f"{{ {_ce(self.conf)} rm -f {name} >/dev/null 2>&1; true; }} && "
                                 f"{shlex.join(argv)} && echo YETO_SSH_STARTED container={name}", timeout=3600)
        text = _text(out.stdout)
        if out.returncode != 0 or "YETO_SSH_STARTED" not in text:
            raise SshSamplerError(f"sampler {m.shard} on {node.name} did not start (exit {out.returncode}): "
                                  f"{(text + _text(out.stderr))[-800:]}")
        _log(f"ssh pool: {m.shard} on {node.name}: {text.strip().splitlines()[-1]}")

    def event(self, name: str, **detail: Any) -> None:
        """One event of the pool's event log (and its ``event_command`` hook), as the NC route writes them."""
        from .nc import cli  # noqa: PLC0415

        hook = island_status.command_hook(self.cfg["event_command"]) if self.cfg.get("event_command") else None
        try:
            island_status.EventLog(path=cli.pool_events_path(self.cfg), run=self.run_id, log=_log, hook=hook)(
                name, **detail)
        except (OSError, KeyError, ValueError) as error:  # an event never stops a start
            _log(f"ssh pool: event {name} not written: {error}")

    def image_for(self, node: Node) -> str:
        """The image a node runs: ``container.image`` (``auto``: the node's inventory
        image, else the catalog's for what the spec asks), else the node's image."""
        return node_image(self.conf, node, spec_of(self.cfg))

    def start_node_runtime(self, route: Any, node: Node, m, shards: list[str], spec: Mapping[str, Any], *,
                           python: str, env_file: str | None) -> str:
        """Write ``m``'s assignment, put its token and the node's profile next to the spec,
        and return the node runtime's start line."""
        from yeto.node.route import runtime_command  # noqa: PLC0415

        p = node_paths(self.conf.workdir, m.shard)
        profile = self.node_profile(node)
        # provider.env goes in the assignment's environment (applied last by the runtime, as the
        # route applies it last); PYTHONPATH stays the route's (its start line prepends the tree)
        env = {k: v for k, v in self.conf.env.items() if k != "PYTHONPATH"}
        # no cards in the assignment: the route's own card variables (device_env: numbered as the
        # container or the venv sees them) reach the sampler through the runtime's environment
        assignment, token = route.assign(m, shards, profile, spec, spec_paths="host", python=python, env=env)
        token_file, profile_file = f"{p['dir']}/{m.shard}.token", f"{p['dir']}/{m.shard}.profile.json"
        self.put(node, token, token_file, mode="600")
        self.put(node, route.profile_file_text(profile), profile_file, mode="644")
        _log(f"ssh pool: {m.shard} on {node.name}: node runtime, assignment {assignment.ticket} "
             f"generation {assignment.generation}")
        return runtime_command(python, assignment.ticket, route.store, env_file=env_file, token_file=token_file,
                               profile_file=profile_file, log="-",
                               child_pid_file=p["pid"] if self.conf.runtime == "venv" else None)

    def restart(self, m, shards: list[str]) -> None:
        self.start(m, shards)

    def ready(self, m) -> bool:
        return self.probe(m).joined

    def state_command(self, shard: str) -> str:
        p = node_paths(self.conf.workdir, shard)
        if self.conf.runtime == "venv":
            return venv_state_command(p)
        return container_state_command(self.conf, container_name(shard))

    def logs_command(self, shard: str, tail: int = 2000) -> str:
        p = node_paths(self.conf.workdir, shard)
        if self.conf.runtime == "venv":
            return f"tail -n {int(tail)} {p['log']} 2>/dev/null"
        return f"{_ce(self.conf)} logs --tail {int(tail)} {container_name(shard)} 2>&1"

    def probe(self, m) -> island_status.Probe:
        from .nc import elastic  # noqa: PLC0415

        try:
            node = self.provider.node_of_ticket(m.ticket)
        except KeyError:
            return island_status.Probe(node=island_status.PROBE_UNREACHABLE)
        tape = node_paths(self.conf.workdir, m.shard)["tape"]
        script = elastic.probe_script(container_name(m.shard), tape[:-len(".jsonl")] + "*.jsonl",
                                      state_script=self.state_command(m.shard),
                                      logs_script=self.logs_command(m.shard))
        try:
            out = self.ssh(node, script, timeout=90)
        except subprocess.TimeoutExpired:
            return island_status.Probe(node=island_status.PROBE_UNREACHABLE)
        return island_status.Probe.parse(_text(out.stdout), out.returncode)

    def logs(self, m, *, tail: int = 200, save: bool = True) -> str:
        """The sampler's log tail, copied to ``<state dir>/ssh-pool-logs/<shard>.log``."""
        node = self.provider.node_of_ticket(m.ticket)
        text = _text(self.ssh(node, self.logs_command(m.shard, tail), timeout=120).stdout)
        if save:
            d = self.state_dir / "ssh-pool-logs"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{m.shard}.log").write_text(text)
        return text

    def _stop_on(self, node: Node, shard: str, grace_s: int | None = None) -> str:
        return _text(self._stop_result(node, shard, grace_s).stdout)

    def _stop_result(self, node: Node, shard: str, grace_s: int | None = None) -> Any:
        grace = int(self.conf.grace_s if grace_s is None else grace_s)
        p = node_paths(self.conf.workdir, shard)
        if self.conf.runtime == "venv":
            cmd = f"{venv_stop_command(p, grace)}; tail -n 300 {p['log']} > {p['logs']}/{shard}.last.log 2>/dev/null"
        else:
            cmd = container_stop_command(self.conf, p, container_name(shard), grace)
        return self.ssh(node, cmd, timeout=grace + 180)

    def _owed_stops(self, node: Node, m) -> None:
        """Stop the samplers whose stop never reached this node (its slot's lost members)."""
        slot = int(self.provider._rows()[m.ticket]["slot"])  # noqa: SLF001 - our own provider's record
        for ticket, shard in self.provider.owed_stops(slot):
            if ticket == m.ticket:
                continue
            out = self._stop_result(node, shard)
            if out.returncode == SSH_UNREACHABLE or "stopped" not in _text(out.stdout):
                raise SshSamplerError(f"node {node.name}: the stop of {shard} (lost with {ticket}) owed there did "
                                      f"not go through (exit {out.returncode}): {_text(out.stderr)[-300:]}")
            self.provider.stop_done(ticket)
            _log(f"ssh pool: {node.name}: stopped {shard} of lost {ticket} before {m.shard} starts there")

    def stop(self, m, *, grace_s: int | None = None) -> str:
        node = self.provider.node_of_ticket(m.ticket)
        try:
            self.logs(m, tail=2000)
        except Exception:  # noqa: BLE001 - the log copy is best effort; the stop is not
            pass
        return self._stop_on(node, m.shard, grace_s)

    # -- the work contract
    def data_path(self) -> Any:
        from .nc import cli  # noqa: PLC0415

        if self.conf.store:
            raise SshSamplerError("a node-local store (provider.store) has no pool fence; the sampler is stopped")
        return cli.pool_data_path(self.cfg, self._fence_store, lambda: None)

    def _fence_store(self) -> Any:
        from .nc import cli  # noqa: PLC0415

        return pool_fence_store(self.cfg, error=cli.NcError)

    def fence(self, m) -> Any:
        if self.conf.store:
            return {"fenced_by_stop": self.stop(m)}
        return self.data_path().fence_shard(m.shard, by=m.ticket)

    def fence_incarnation(self, m) -> Any:
        if self.conf.store:
            return {"fenced_by_stop": self.stop(m)}
        return self.data_path().fence_incarnation(m.shard, by=m.ticket)

    def drain(self, m) -> None:
        """Stop the sampler and release the ticket. A node the pool saw unreachable, or that
        the stop did not reach, is marked lost (:meth:`SshNodeProvider.mark_lost`): its slot
        is held while a spare is free, and a stop that did not get there is owed."""
        lost = getattr(m, "unreachable_since", None) is not None
        stopped = not m.shard
        if m.shard:
            node = self.provider.node_of_ticket(m.ticket)
            try:
                self.logs(m, tail=2000)
            except Exception:  # noqa: BLE001 - the log copy is best effort; the stop is not
                pass
            try:
                out = self._stop_result(node, m.shard)
                text = _text(out.stdout)
                stopped = out.returncode != SSH_UNREACHABLE and "stopped" in text
                lost = lost or out.returncode == SSH_UNREACHABLE
                _log(f"ssh pool: stop {m.shard}: {text.strip()[-300:] or _text(out.stderr).strip()[-300:]}")
            except subprocess.TimeoutExpired:
                lost = True
                _log(f"ssh pool: stop {m.shard} timed out; the ticket is ended, check the node")
        if lost or not stopped:
            self.provider.mark_lost(m.ticket, stopped=stopped)
            if lost:
                _log(f"ssh pool: node {self.provider.node_of_ticket(m.ticket).name} lost with {m.ticket}: held "
                     f"{self.conf.lost_node_hold_s:.0f} s while a spare node is free"
                     + ("" if stopped else f"; the stop of {m.shard} is owed to its next start"))
        self.provider.release(m.account, m.ticket)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _log(message: str) -> None:
    print(message, flush=True)


# -- registry hooks -------------------------------------------------------------


def registry_factory(cfg: dict[str, Any], run_id: str, section: dict[str, Any]) -> SshNodeProvider:  # noqa: ARG001
    return pool_ops(cfg).provider


def pool_ops(cfg: dict[str, Any]) -> SshSamplerPoolOps:
    return SshSamplerPoolOps(cfg)


def offline_problems(cfg: dict[str, Any]) -> list[str]:
    """``pool --dry-run`` checks (no SSH)."""
    try:
        conf = parse_config(cfg)
    except (ValueError, TypeError, KeyError, OSError) as error:
        return [str(error)]
    problems = []
    if cfg.get("route", "ssh") != "ssh":
        problems.append("ssh provider: route must be ssh (the default)")
    if cfg.get("model_pull") or cfg.get("node_hooks") or cfg.get("sampler_route"):
        problems.append("ssh provider: model_pull / node_hooks / sampler_route are routes of rented nodes; "
                        "put the model on the node (inventory mounts) and use runtime container|venv")
    if conf.store and int(cfg.get("target", 1)) != 1:
        problems.append("ssh provider: a node-local store (provider.store) serves one sampler; target must be 1")
    if not conf.store and cfg.get("data_path", "r2") == "r2" and not cfg.get("r2_env_file"):
        problems.append("ssh provider: data_path r2 needs r2_env_file (or set provider.store, or data_path syncer)")
    problems += spec_node_problems(conf, spec_of(cfg))
    problems += image_problems(conf, spec_of(cfg))
    return problems


def spec_node_problems(conf: SshPoolConf, spec: Mapping[str, Any]) -> list[str]:
    """``pool --dry-run``: the spec's per-platform keys fit every node (a pool's nodes run
    one spec): ``platform`` and ``engine.server.device_family`` name the node's family,
    ``engine.server.card_model`` its card. Absent, placeholder or unknown values are not checked."""
    from ..gpu_spec import UnknownCard, UnknownFamily, canonical_card, unified_family  # noqa: PLC0415

    server = (spec.get("engine") or {}).get("server") or {}
    out = []
    for node in conf.nodes:
        profile = profile_of(node)
        for key, value in (("platform", spec.get("platform")), ("engine.server.device_family",
                                                                server.get("device_family"))):
            if not isinstance(value, str) or not value or "<" in value:
                continue
            try:
                family = unified_family(value)
            except UnknownFamily:
                continue
            if family != profile.family:
                out.append(f"ssh provider: node {node.name} is a {profile.family} node, the spec's {key} is {value}")
        card = server.get("card_model")
        if isinstance(card, str) and card and "<" not in card:
            try:
                if canonical_card(card) != node.card:
                    out.append(f"ssh provider: node {node.name} has {node.card} cards, the spec's "
                               f"engine.server.card_model is {card}")
            except UnknownCard:
                pass
    return out


def spec_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The pool's sampler spec file as written (``{}`` when it cannot be read: the
    pool's own checks report that)."""
    try:
        return json.loads(Path(os.path.expanduser(str(cfg["spec"]))).read_text())
    except (KeyError, OSError, ValueError, TypeError):
        return {}


def node_image(conf: SshPoolConf, node: Node, spec: Mapping[str, Any]) -> str:
    """The image ``node`` runs (container runtime): ``container.image`` unless it is
    ``auto``; then the node's inventory image, else the engine image catalog's choice
    for what ``spec`` asks (raises ``ImageIncompatible`` when none serves it)."""
    assert conf.container is not None
    if conf.container.image and conf.container.image != AUTO_IMAGE:
        return conf.container.image
    if node.image:
        return node.image
    needs = engine_images.spec_needs(spec)
    return engine_images.resolve(profile_of(node), needs.pop("model"), needs.pop("engine_pref"),
                                 **needs).image.pull_ref  # type: ignore[union-attr]


def image_problems(conf: SshPoolConf, spec: Mapping[str, Any]) -> list[str]:
    """``pool --dry-run``: every node's image serves the spec (a catalog image is held to
    the node's family, card architecture and HBM and to the spec's model, reload mode,
    precision and rollout fields; a site image is the site's)."""
    if conf.runtime != "container" or conf.container is None:
        return []
    needs = engine_images.spec_needs(spec)
    model, pref = needs.pop("model"), needs.pop("engine_pref")
    out = []
    for node in conf.nodes:
        try:
            image = node_image(conf, node, spec)
            engine_images.resolve(profile_of(node), model, pref, image=image, **needs)
        except engine_images.ImageIncompatible as error:
            out.append(f"ssh provider: node {node.name}: {error}")
    return out


def required_pool_keys(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """The pool keys a ``kind: ssh`` pool needs (``r2_env_file`` only on the r2 data path)."""
    section = cfg.get("provider") or {}
    if section.get("store") or cfg.get("data_path", "r2") != "r2":
        return ("spec", "run_prefix")
    return ("spec", "run_prefix", "r2_env_file")
