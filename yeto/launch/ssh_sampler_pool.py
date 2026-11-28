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

Node health: on the node runtime, a node of a family with a health checker (cuda,
:mod:`yeto.hw.node_health`) is checked every minute while its sampler runs; the probe
reads the newest check (``<workdir>/sampler/<shard>.health.json``) in its own ssh call.
Two bad checks in a row read ``lost``: the pool takes the node for lost (fence, then a
replacement), and the node's slot is held as for a node that stopped answering.

Leases across runs: an inventory with a top-level ``lease_store`` (:mod:`.node_lease`)
leases its nodes to one run at a time. A slot whose node another run holds is not free
(the holder is reported once per change, event ``node_leased_elsewhere``); an order
takes the node's lease first (refused: ``NodeLeased``, the pool tries the next slot);
each ticket read renews the pool's leases; a released ticket gives its node's lease up
unless a stop is still owed there. A lease lost to another run ends the member (lost).
Without ``lease_store`` none of this runs.

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
import logging
import os
import shlex
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import engine_images, ssh_transport
from .node_inventory import (
    NPU_HOST_DEVICES,  # noqa: F401 - re-exported (they lived here)
    NPU_HOST_MOUNTS,  # noqa: F401
    InventoryError,
    Node,
    NodeProfile,
    lease_store_of,
    load_profiles,
    nodes_by_name,
    parse_node_health,
    profile_of,
)
from .node_lease import SAMPLER_MARKER
from .pool import status as island_status
from .pool.provider import provider_caps
from .pool.status import LeaseStatus
from yeto.node.store import STORE_URI_SHELL, pool_fence_store, reclaim_marker_env, sampler_route_env
from yeto.node.transport import ReachError, node_ssh_options

KIND = "ssh"
SSH_UNREACHABLE = 255  # ssh's own failure (no connection, no session): the node was not reached
LEASE_READ_CACHE_S = 15.0  # a node's lease holder read for a free-slot check is reused this long
REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = SAMPLER_MARKER  # prefix of a start's marker (start_marker): run, shard, ticket (node_lease's probe)
MARK_ENV = "YETO_SSH_SAMPLER_MARK"  # the sampler process carries its start's marker in this variable
RUNTIMES = ("container", "venv")
ENGINES = ("podman", "docker")
PULL_POLICIES = ("missing", "always", "never")
CONFIG_KEYS = frozenset({"kind", "inventory", "nodes", "workdir", "runtime", "container", "venv", "env",
                         "devices", "store", "preflight", "grace_s", "price_usd_h", "start_timeout_s",
                         "lost_node_hold_s", "node_runtime", "node_health"})
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


class PromptsConfigError(SshSamplerError):
    config_error = True  # the pool config's prompts_file / prompts.path: no retry fixes it


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
    # the node runtime's health check (yeto.hw.node_health): "auto" / "off" for every node of
    # the pool; None: each node's inventory node_health (not declared: auto)
    node_health: str | None = None
    # the inventory's top-level lease_store (yeto.launch.node_lease); None: nodes not leased
    lease_store: str | None = None
    # the site image catalog a ``container.image: auto`` node without an inventory image reads
    # (the inventory's top-level site_images:, else ~/.yeto/site-images.yaml; yeto.launch.site_images)
    site_images: str | None = None


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
    lease_store = lease_store_of(inventory)
    from .site_images import inventory_catalog_path  # noqa: PLC0415

    site_catalog = str(inventory_catalog_path(section.get("inventory")))
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
        other = [f"{n.name} (roles: {', '.join(n.roles)})" for n in nodes if n.roles and "sampler" not in n.roles]
        if other:  # a node with roles serves those roles only (no roles: any)
            raise ValueError(f"ssh provider: nodes {'; '.join(other)} have no sampler role in the inventory; add "
                             "sampler to their roles, or name sampler nodes in provider.nodes")
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
    try:  # provider.node_health: auto | off for the pool's nodes (absent: each node's inventory value)
        health = parse_node_health(section.get("node_health"), "ssh provider")
    except InventoryError as error:
        raise ValueError(str(error)) from None
    return SshPoolConf(nodes=tuple(nodes), workdir=workdir.rstrip("/"), runtime=runtime, container=container,
                       venv=venv, env={str(k): str(v) for k, v in env.items()}, devices=devices,
                       store=str(store) if store else None, min_free_gib=float(pre.get("min_free_gib", 20)),
                       require_idle=(bool(pre["require_idle"]) if pre.get("require_idle") is not None else None),
                       min_cards=int(pre["min_cards"]) if pre.get("min_cards") is not None else None,
                       grace_s=int(section.get("grace_s", 120)),
                       price_usd_h=float(price) if price is not None else None, lost_node_hold_s=float(hold),
                       node_runtime=node_runtime, node_health=health, lease_store=lease_store,
                       site_images=site_catalog)


# -- node shell -----------------------------------------------------------------


def node_paths(workdir: str, shard: str) -> dict[str, str]:
    s = f"{workdir}/sampler"
    return {"dir": s, "spec": f"{s}/{shard}.json", "script": f"{s}/{shard}.start.sh", "pid": f"{s}/{shard}.pid",
            "pgid": f"{s}/{shard}.pgid", "exit": f"{s}/{shard}.exit", "log": f"{workdir}/logs/{shard}.log",
            "tape": f"{s}/events-{shard}.jsonl", "logs": f"{workdir}/logs", "marker": f"{s}/{shard}.marker"}


HEALTH_MARK = "YETO_NODE_HEALTH"


def health_path(workdir: str, shard: str) -> str:
    """Where the node runtime writes the sampler's node health check (``yeto.node.runtime``:
    ``<work>/sampler/<shard>.health.json``; the runtime's work is the pool's workdir)."""
    return f"{node_paths(workdir, shard)['dir']}/{shard}.health.json"


def health_probe_command(path: str) -> str:
    """One line ``YETO_NODE_HEALTH <json>`` (empty when the runtime wrote no check yet)."""
    return f"printf '{HEALTH_MARK} %s\\n' \"$(tr -d '\\n' < {shlex.quote(path)} 2>/dev/null)\""


def parse_health_line(text: str) -> dict[str, Any] | None:
    """The health check a probe read (:func:`health_probe_command`); None without one."""
    for line in str(text or "").splitlines():
        if line.startswith(HEALTH_MARK + " "):
            try:
                doc = json.loads(line[len(HEALTH_MARK) + 1:])
            except ValueError:
                return None
            return doc if isinstance(doc, dict) else None
    return None


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
                       host_paths: Sequence[str] | None = None, command: str | None = None,
                       labels: Mapping[str, str] | None = None) -> list[str]:
    """``<engine> run -d``: the node's device flags (the inventory's ``docker_args``,
    else the defaults of its family for ``cards``: :meth:`NodeProfile.devices`),
    ``security_opt``, the workdir and mounts at the same paths as on the host, then
    ``exec`` of the engine sampler as PID 1. ``labels``: ``--label`` flags (a leased
    node's run label, :func:`~.node_lease.run_labels`; none: the argv as before)."""
    c = conf.container
    assert c is not None
    argv = [*(["sudo", "-n"] if c.sudo else []), c.engine, "run", "-d", "--name", name,
            "--network", "host", "--ipc", "host", "--shm-size", c.shm_size]
    for key, value in sorted((labels or {}).items()):
        argv += ["--label", f"{key}={value}"]
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
        # node leases across runs (conf.lease_store; node_lease): opened on first use
        self._leases: Any = None
        self._lease_seen: dict[str, tuple[str | None, float]] = {}  # node -> (holder, read at)
        self._lease_reported: dict[str, str | None] = {}  # node -> the other run last reported
        self.on_event: Callable[..., None] | None = None  # the ops' event tape (SshSamplerPoolOps.event)
        # before an expired lease of another run is taken over: is that run still on the node
        # (node_lease.ssh_probe over the ops' ssh; None: no probe)
        self.lease_probe: Callable[[Any, str], str | None] | None = None
        self._lease_identity: str | None = None  # the pool's launch identity (attach_store: the run store's)
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
            if self.conf.lease_store is not None and self._leases is None:  # one identity on every host
                from .node_lease import store_identity  # noqa: PLC0415

                self._lease_identity = store_identity(objects, f"fleet/pools/{pool}/node-lease-identity.json")
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
        taken = self._take_lease(account)
        try:
            with self._lock:
                rows = self._rows()
                n = 1 + sum(1 for r in rows.values() if int(r["slot"]) == int(account))
                ticket = f"{self.run_id}-{self.node_of_slot(account).name}-{n}"
                rows[ticket] = {"slot": int(account), "node": self.node_of_slot(account).name, "state": "active",
                                "ordered_at": self.clock()}
                self._save(rows)
                return ticket
        except BaseException:
            if taken:
                self._leases.release(self.node_of_slot(account))
            raise

    def status(self, account: int, ticket: str) -> LeaseStatus:  # noqa: ARG002
        row = self._rows().get(ticket)
        if row is None:
            return LeaseStatus("ended", end_reason="lost")
        if row["state"] == "released":
            return LeaseStatus("ended", end_reason="released")
        lost = self._lease_tick(int(row["slot"]))
        if lost is not None:
            return LeaseStatus("ended", end_reason="lost", detail=lost)
        return LeaseStatus("active")

    def release(self, account: int, ticket: str) -> dict[str, Any]:  # noqa: ARG002
        with self._lock:
            rows = self._rows()
            if ticket in rows:
                rows[ticket] = dict(rows[ticket], state="released", released_at=self.clock())
                self._save(rows)
                self._give_lease_back(int(rows[ticket]["slot"]), rows)
        return {"ticket": ticket, "state": "released"}

    def released_by_us(self, ticket: str) -> bool:
        return self._rows().get(ticket, {}).get("state") == "released"

    def node(self, account: int, ticket: str) -> Node:  # noqa: ARG002
        return self.node_of_ticket(ticket)

    def balance(self) -> int | None:
        return None

    def account_free(self, account: int) -> bool:
        rows, now = self._rows(), self.clock()
        if self._busy(rows, account) or self._leased_elsewhere(account):
            return False
        if self._held(rows, account, now) is None:
            return True
        # a lost node: offered again only when no other slot's node is free (no spare left)
        return not any(not self._busy(rows, s) and self._held(rows, s, now) is None
                       and not self._leased_elsewhere(s) for s in self.slots if int(s) != int(account))

    # -- node leases across runs (conf.lease_store, yeto.launch.node_lease) ---------------
    @property
    def leases(self) -> Any:
        """The run's :class:`~.node_lease.NodeLeases` (None: the inventory has no lease_store)."""
        if self.conf.lease_store is None:
            return None
        if self._leases is None:
            from .node_lease import NodeLeases, local_identity  # noqa: PLC0415

            identity = self._lease_identity or local_identity(
                self.state_path.with_name(self.state_path.stem + ".lease-identity.json"))
            pool = getattr(self, "_store_pool", self.run_id)
            where = (f"the run store's fleet/pools/{pool}/node-lease-identity.json" if self._lease_identity
                     else str(self.state_path.with_name(self.state_path.stem + ".lease-identity.json")))
            self._leases = NodeLeases(self.conf.lease_store, f"{self.run_id}/pool", identity=identity,
                                      clock=self.clock, on_event=self._lease_event, probe=self.lease_probe,
                                      identity_path=where)
        return self._leases

    def _lease_event(self, name: str, **detail: Any) -> None:
        if self.on_event is not None:  # the pool's event log (it prints the line too)
            self.on_event(name, **detail)
        else:
            _log(f"ssh pool: {name}: " + " ".join(f"{k}={v}" for k, v in detail.items()))

    def _leased_elsewhere(self, slot: int) -> bool:
        """``slot``'s node is held by another run (read at most every LEASE_READ_CACHE_S);
        a store that does not answer counts as held (never start on a node not confirmed)."""
        if self.conf.lease_store is None:
            return False
        node, now = self.node_of_slot(slot), self.clock()
        seen = self._lease_seen.get(node.name)
        if seen is not None and now - seen[1] < LEASE_READ_CACHE_S:
            holder = seen[0]
        else:
            try:
                holder = self.leases.holder(node)
            except Exception as error:  # noqa: BLE001 - not confirmed free: not offered this time
                _log(f"ssh pool: lease of {node.name} not readable ({error!r}): the node is not offered")
                return True
            self._lease_seen[node.name] = (holder, now)
        other = holder if holder not in (None, self.leases.holder_id) else None
        if self._lease_reported.get(node.name) != other:
            self._lease_reported[node.name] = other
            if other is not None:
                self._lease_event("node_leased_elsewhere", node=node.name, holder=other)
        return other is not None

    def _take_lease(self, slot: int) -> bool:
        """Before an order on ``slot``: take its node's lease (no lease_store: nothing).
        Raises ``NodeLeased`` when another run holds the node."""
        leases = self.leases
        if leases is None:
            return False
        from .node_lease import NodeLeased  # noqa: PLC0415

        node = self.node_of_slot(slot)
        self._lease_seen.pop(node.name, None)
        try:
            got = leases.acquire(node)
        except Exception as error:  # noqa: BLE001 - the store: no lease confirmed, a refusal (no failure)
            self._lease_event("node_lease_store_error", node=node.name, error=repr(error)[:200])
            raise NodeLeased(f"ssh inventory: the lease of node {node.name} is not confirmed ({error!r})"[:300],
                             {node.name: None}) from error
        if not got.ok:
            same = leases.same_run_conflict(got.holder)
            if same:  # this pool under another identity: never free by waiting
                self._lease_event("node_lease_same_run", node=node.name, holder=got.holder, why=same[:300])
                raise SshSamplerError(f"ssh inventory: node {node.name} is {same}")
            why = got.blocked or f"node {node.name} is held by run {got.holder}"
            raise NodeLeased(f"ssh inventory: {why}", {node.name: got.holder})
        return True

    def _lease_tick(self, slot: int) -> str | None:
        """On a ticket read: renew the pool's leases that are due; why the member on
        ``slot`` lost its node to another run (None: it did not, or no lease_store). A
        store that does not answer ends nothing."""
        leases = self.leases
        if leases is None:
            return None
        node = self.node_of_slot(slot)
        try:
            if node.name not in leases.held() and node.name not in leases.lost() and not leases.renew(node):
                # an adopted ticket (a restarted controller): its node is another run's now
                return f"node {node.name}: its lease is held by run {leases.holder(node)}"
            leases.renew_due()
        except Exception as error:  # noqa: BLE001 - renewed at a later read; the lease still runs
            _log(f"ssh pool: node leases not renewed ({error!r})")
            return None
        lost = leases.lost()
        if node.name in lost:
            return f"node {node.name}: its lease is held by run {lost[node.name]} now"
        return None

    def _give_lease_back(self, slot: int, rows: Mapping[str, Mapping[str, Any]]) -> None:
        """Release ``slot``'s node lease once no ticket of this pool is active there and no
        stop is owed there (a sampler that may still run keeps the node this run's)."""
        if self._leases is None:
            return
        if any(int(r["slot"]) == int(slot) and (r["state"] == "active" or r.get("stop_owed"))
               for r in rows.values()):
            return
        node = self.node_of_slot(slot)
        self._leases.release(node)
        self._lease_seen.pop(node.name, None)

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
                self._give_lease_back(int(rows[ticket]["slot"]), rows)

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
        self.provider.on_event = self.event  # node lease reports (lease_store) on the pool's tape
        if self.conf.lease_store is not None:  # a takeover first asks the node whether the old run still runs
            from .node_lease import ssh_probe  # noqa: PLC0415

            sudo = bool(self.conf.container is not None and self.conf.container.sudo)
            self.provider.lease_probe = ssh_probe(lambda node, script: self.ssh(node, script, timeout=120), sudo=sudo)
        self._shard_of: dict[str, str] = {}
        self._node_route: Any = None
        self._health_generation: dict[str, int] = {}  # ticket -> the assignment its health check is of
        self._health_lost: set[str] = set()  # tickets whose node's health read lost (drain holds its slot)
        self._images: dict[str, Any] | None = None  # node -> site_images.Choice, fixed at the first start
        self._images_lock = threading.Lock()  # parallel starts (IslandPool) choose the images once

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

    def upload_prompts(self, node: Node, spec: Mapping[str, Any]) -> dict[str, Any]:
        """The pool's ``prompts_file`` at the host path behind the spec's ``prompts.path``
        on ``node`` (:func:`prompts_host_path`), checked there by size and sha256 against
        the local file: a copy that matches already is kept (``uploaded`` False), else the
        file is streamed there (a temp name of this shell, then renamed) and checked again.
        :class:`PromptsConfigError` (not retried) when the sampler would not see the path
        or the local file is missing; SshSamplerError when the copy fails or does not match."""
        from yeto.node.transport import size_sha256_command  # noqa: PLC0415

        local = Path(os.path.expanduser(str(self.cfg["prompts_file"])))
        prompts = spec.get("prompts") if isinstance(spec.get("prompts"), Mapping) else {}
        remote = str(prompts.get("path") or "")
        host, _where = prompts_host_path(self.conf, node, remote)
        if host is None:
            raise PromptsConfigError(f"prompts_file {local}: the sampler spec's prompts.path "
                                     f"{remote or '(unset)'!r} is under no mount or workdir of {node.name}: the "
                                     "sampler would not see it")
        try:
            size, digest = file_size_sha256(local)
        except OSError as error:
            raise PromptsConfigError(f"prompts_file {local}: {error.strerror or error}") from None
        copy = {"bytes": size, "sha256": digest}

        def matches() -> tuple[bool, str]:
            out = self.ssh(node, size_sha256_command(host), timeout=120)
            have = next((ln.split()[1:] for ln in _text(out.stdout).splitlines() if ln.startswith("YETO_PROMPTS ")), [])
            return have == [str(size), digest], " ".join(have) or "unreadable"

        same, _have = matches()
        if same:
            _log(f"ssh pool: {node.name}: prompts_file {local} is at {host} already ({size} bytes, sha256 "
                 f"{digest[:12]}): kept")
            return {"path": remote, "host": host, "uploaded": False, **copy}
        out = self.ssh(node, f"mkdir -p {shlex.quote(str(Path(host).parent))}", timeout=60)
        if out.returncode != 0:
            raise SshSamplerError(f"prompts upload to {node.name}: {_text(out.stderr)[-300:]}")
        q = shlex.quote(host)
        with local.open("rb") as stream:  # streamed: a large file is not held in memory
            out = self.ssh(node, f"umask 077; t={q}.tmp.$$; cat > \"$t\" && chmod 644 \"$t\" && mv -f \"$t\" {q} "
                                 "|| { rm -f \"$t\"; exit 1; }", stdin=stream, timeout=upload_timeout_s(size))
        if out.returncode != 0:
            raise SshSamplerError(f"prompts upload {host} to {node.name} (exit {out.returncode}): "
                                  f"{_text(out.stderr)[-300:]}")
        same, have = matches()
        if not same:
            raise SshSamplerError(f"{node.name}: prompts file {host} does not match {local} after the upload "
                                  f"(node size/sha256 {have}, local {size} {digest[:12]})")
        _log(f"ssh pool: {node.name}: prompts_file {local} -> {host} ({size} bytes, sha256 {digest[:12]}) checked")
        return {"path": remote, "host": host, "uploaded": True, **copy}

    def upload_code(self, node: Node) -> str:
        """``git archive`` (gzip: a slow launcher link) of ``yeto/`` at ``code_ref`` into
        ``<workdir>/yeto/<sha>`` (idempotent: a tree already there is kept)."""
        from yeto.node.code import code_sha  # noqa: PLC0415

        sha = code_sha(self.code_ref, repo=REPO_ROOT, run=self._run)
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
        if self.cfg.get("prompts_file"):  # at every start, as the rented-node route: copied, then checked there
            self.upload_prompts(node, spec)
        route = self.node_route_for(node)
        if route is None:
            self.put(node, json.dumps(spec, indent=1), p["spec"])
        env_file = None
        if self._uses_r2():
            env_file = f"{self.conf.workdir}/sampler/.store.env"
            self.put(node, Path(os.path.expanduser(str(self.cfg["r2_env_file"]))).read_text(), env_file, mode="600")
        env = self.sampler_env(node, m)
        labels = None
        if self.conf.lease_store is not None:  # the takeover probe finds this run's containers by label
            from .node_lease import RUN_LABEL_ENV, run_labels  # noqa: PLC0415

            labels = run_labels(self.run_id)
            env[RUN_LABEL_ENV] = self.run_id
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
                                      host_paths=facts.get("host_paths"), command=exec_line, labels=labels)
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
        image, else the site image catalog's, else the engine image catalog's for what the
        spec asks), else the node's image. An image the pool chose itself is chosen once
        for every node (:func:`pool_image_choices`, what the dry-run prints) and kept in
        the pool's state (:meth:`images_path`): a replacement node and a restarted pool
        controller start the same image, the catalog is not read again."""
        images = self._images
        if images is None:
            with self._images_lock:  # double-checked: one start chooses, the others take its choice
                if self._images is None:
                    self._images = self._pin_images()
                images = self._images
        choice = images.get(node.name)
        if choice is None:
            return node_image(self.conf, node, spec_of(self.cfg))
        if isinstance(choice, Exception):
            raise choice
        return str(choice.ref)

    def images_path(self) -> Path:
        return self.state_dir / f"ssh-pool-{self.run_id}-images.json"

    def _pin_images(self) -> dict[str, Any]:
        from . import site_images  # noqa: PLC0415

        spec = spec_of(self.cfg)
        choices = pool_image_choices(self.conf, spec)
        needs = engine_images.spec_needs(spec)
        key = {"model": needs.get("model"), "engine": needs.get("engine_pref"), "catalog": self.conf.site_images}
        path = self.images_path()
        try:
            kept = json.loads(path.read_text())
        except (OSError, ValueError):
            kept = None
        if isinstance(kept, dict) and kept.get("key") == key:
            for name, row in (kept.get("images") or {}).items():
                if name in choices and not _explicit(choices[name]):
                    choices[name] = site_images.Choice(str(row["ref"]), str(row["source"]),
                                                       why=f"kept from the pool's start ({path.name})")
        pinned = {name: {"ref": c.ref, "source": c.source} for name, c in choices.items()
                  if not isinstance(c, Exception) and not _explicit(c)}
        if pinned:
            try:
                import tempfile  # noqa: PLC0415

                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
                try:
                    with os.fdopen(fd, "w") as handle:
                        handle.write(json.dumps({"key": key, "images": pinned}, indent=1, sort_keys=True))
                    os.replace(tmp, path)
                except BaseException:
                    Path(tmp).unlink(missing_ok=True)
                    raise
            except OSError as error:  # the pool runs; a restarted controller chooses again
                _log(f"ssh pool: the chosen images were not kept in {path}: {error}")
            for name, c in sorted(choices.items()):
                if not isinstance(c, Exception) and not _explicit(c):
                    _log(f"ssh pool: node {name}: {c.line('sampler')}")
        return choices

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
        self._health_generation[m.ticket] = int(assignment.generation)
        if self.health_checked(node):  # no check of an earlier sampler on this node is read as this one's
            try:  # the start goes on: the probe ignores a check of another ticket or assignment anyway
                self.ssh(node, "rm -f " + shlex.quote(health_path(self.conf.workdir, m.shard)), timeout=60)
            except Exception as error:  # noqa: BLE001 - observation only: the start does not depend on it
                _log(f"ssh pool: {m.shard} on {node.name}: old health check not removed: {error}")
        token_file, profile_file = f"{p['dir']}/{m.shard}.token", f"{p['dir']}/{m.shard}.profile.json"
        self.put(node, token, token_file, mode="600")
        # provider.node_health wins over the node's inventory value (the runtime reads the profile's)
        health = {"node_health": self.conf.node_health} if self.conf.node_health is not None else {}
        self.put(node, route.profile_file_text(profile, **health), profile_file, mode="644")
        _log(f"ssh pool: {m.shard} on {node.name}: node runtime, assignment {assignment.ticket} "
             f"generation {assignment.generation}")
        return runtime_command(python, assignment.ticket, route.store, env_file=env_file, token_file=token_file,
                               profile_file=profile_file, log="-",
                               child_pid_file=p["pid"] if self.conf.runtime == "venv" else None,
                               health_every_s=0 if self.health_mode(node) == "off" else None)

    def restart(self, m, shards: list[str]) -> None:
        self.start(m, shards)

    def ready(self, m) -> bool:
        return self.probe(m).joined

    def serving_evidence(self, m) -> bool:
        """The run store says ``m``'s sampler serves: its node runtime wrote the ready record of
        the ticket's current assignment, its heartbeat is fresh and no exit is recorded
        (``yeto.node.records``). Durable, unlike the probe's JOIN grep (the container log's
        last 2000 lines, the node's tape): a control plane that took the pool over reads it
        too. False: no node runtime / store, or not serving."""
        from yeto.node.records import node_status  # noqa: PLC0415

        try:
            node = self.provider.node_of_ticket(m.ticket)
        except KeyError:
            return False
        route = self.node_route_for(node)
        if route is None:
            return False
        return node_status(route.objects, m.ticket).state == "ready"

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
        health = self.health_checked(node)
        if health:
            script += "; " + health_probe_command(health_path(self.conf.workdir, m.shard))
        try:
            out = self.ssh(node, script, timeout=90)
        except subprocess.TimeoutExpired:
            return island_status.Probe(node=island_status.PROBE_UNREACHABLE)
        probe = island_status.Probe.parse(_text(out.stdout), out.returncode)
        if probe.node == island_status.PROBE_OK and probe.sampler == "running" and not probe.joined:
            try:  # the JOIN line left the log tail / the tape is elsewhere: the store's ready record
                if self.serving_evidence(m):
                    probe = replace(probe, joined=True)
            except Exception as error:  # noqa: BLE001 - observation only: the probe stands as read
                _log(f"ssh pool: {m.shard}: node records not read: {error!r}"[:300])
        if health and probe.node == island_status.PROBE_OK:
            from yeto.hw.node_health import fresh  # noqa: PLC0415

            doc = parse_health_line(_text(out.stdout))
            want = self._health_generation.get(m.ticket)
            if doc is not None and doc.get("ticket") in (None, m.ticket) \
                    and (want is None or doc.get("generation") in (None, want)):
                probe = replace(probe, health=fresh(doc, time.time()))  # stale: a read failure, not lost
                if island_status.health_lost(probe) is not None:
                    self._health_lost.add(m.ticket)
        return probe

    def health_checked(self, node: Node) -> bool:
        """The node runtime checks ``node``'s health (it starts the sampler, and its family has
        a checker: :mod:`yeto.hw.node_health`)."""
        from yeto.hw.node_health import checker_for  # noqa: PLC0415

        return bool(self.conf.node_runtime and (self.conf.store or self._uses_r2())
                    and checker_for(node.device_family).enabled and self.health_mode(node) != "off")

    def health_mode(self, node: Node) -> str | None:
        """``provider.node_health``, else the node's inventory ``node_health`` (None: neither)."""
        return health_mode(self.conf, node)

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
        # a node that stopped answering, or whose health read lost: its slot is held (mark_lost)
        lost = getattr(m, "unreachable_since", None) is not None or m.ticket in self._health_lost
        self._health_lost.discard(m.ticket)
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
    health = health_summary(conf)
    if health is not None:
        _log(f"ssh pool: node health: {health}")
    return problems


def health_mode(conf: SshPoolConf, node: Node) -> str | None:
    """``provider.node_health``, else the node's inventory ``node_health`` (None: neither)."""
    return conf.node_health if conf.node_health is not None else getattr(node, "node_health", None)


def health_summary(conf: SshPoolConf) -> str | None:
    """``on (<family>)`` / ``off`` when the config or the inventory declares node_health
    (None otherwise: a DEBUG log says it, the dry run prints nothing). The check runs on
    the node runtime (``provider.node_runtime``)."""
    from yeto.hw.node_health import checker_for  # noqa: PLC0415

    modes = [health_mode(conf, n) for n in conf.nodes]
    on = sorted({n.device_family for n, mode in zip(conf.nodes, modes)
                 if mode != "off" and checker_for(n.device_family).enabled and conf.node_runtime})
    state = f"on ({', '.join(on)})" if on else "off"
    if all(mode is None for mode in modes):
        logging.getLogger(__name__).debug("ssh pool: node health: %s (not declared)", state)
        return None
    return state


def lease_notes(cfg: dict[str, Any]) -> list[str]:
    """``pool --dry-run``: how the pool's nodes are leased across runs: the inventory's
    ``lease_store``, or the shared hint that they are not (once per process)."""
    try:
        conf = parse_config(cfg)
    except (ValueError, TypeError, KeyError, OSError):
        return []  # offline_problems reports it
    if conf.lease_store is None:
        from .node_lease import no_lease_hint  # noqa: PLC0415

        out: list[str] = []
        no_lease_hint(emit=out.append)
        return out
    return [f"node leases: lease_store {conf.lease_store}: a node another run holds is skipped; an order takes "
            "the node's lease first (no SSH or store call in a dry run)"]


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
        have = int(node.cards_per_node)
        beyond = [d for d in conf.devices or () if d >= have]
        if beyond:
            out.append(f"ssh provider: node {node.name} has {have} cards (indexes 0-{have - 1}); provider.devices "
                       f"{beyond} are not on it")
        try:
            tp = spec_tp(spec)
        except ValueError as error:
            out.append(f"ssh provider: {error}")
            continue
        cards = min(len(conf.devices), have) if conf.devices else have
        if tp is not None and tp > cards:
            where = (f"provider.devices {list(conf.devices)} give it {cards}" if conf.devices
                     else f"it has {cards}")
            out.append(f"ssh provider: node {node.name}: the spec's engine tensor parallel size is {tp} cards, "
                       f"{where}; lower engine.options.tp or give the pool more cards of the node")
    return out


# the adapters' own ``tp`` defaults (VllmOptions, SGLangOptions; the launch layer imports no
# adapter: tests/test_ssh_pool_checks.py holds the two equal)
ADAPTER_DEFAULT_TP = {"vllm-server": 4, "sglang-server": 8}


def spec_tp(spec: Mapping[str, Any]) -> int | None:
    """The cards one sampler server spans: ``engine.options.tp`` (a whole number, or its
    decimal string), else the adapter's default (vllm-server, sglang-server); None for
    another adapter. ValueError when ``tp`` is no whole number >= 1."""
    engine = spec.get("engine") if isinstance(spec.get("engine"), Mapping) else {}
    options = engine.get("options") if isinstance(engine.get("options"), Mapping) else {}
    tp = options.get("tp")
    if tp is None:
        tp = ADAPTER_DEFAULT_TP.get(str(engine.get("adapter")))
        if tp is None:
            return None
    if isinstance(tp, str) and tp.strip().isdigit():
        tp = int(tp.strip())
    if isinstance(tp, bool) or not isinstance(tp, int) or tp < 1:
        raise ValueError(f"engine.options.tp must be a whole number of cards >= 1, not {tp!r}")
    return tp


PROMPTS_WORKDIR, PROMPTS_MOUNT, PROMPTS_HOST = "workdir", "mount", "host"
UPLOAD_BYTES_PER_S = 10 * 1024 * 1024  # the slowest link a prompts upload is given time for
UPLOAD_TIMEOUT_MAX_S = 1800


def upload_timeout_s(size: int) -> float:
    """The time a ``size``-byte upload is given: 60 s plus the bytes at 10 MiB/s, at most 30 min."""
    return min(float(UPLOAD_TIMEOUT_MAX_S), 60.0 + size / UPLOAD_BYTES_PER_S)


def file_size_sha256(path: Path) -> tuple[int, str]:
    """``(bytes, sha256 hex)`` of a local file, read in chunks."""
    import hashlib  # noqa: PLC0415

    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def prompts_host_path(conf: SshPoolConf, node: Node, path: str) -> tuple[str | None, str | None]:
    """``(host path, where)`` of the sampler's ``prompts.path`` on ``node``: under the
    pool's workdir (``workdir``), under a mount (``mount``: the inventory's or the pool's
    ``container.mounts``, the path on the host behind it), or (runtime venv, which reads
    the host itself) elsewhere on the host (``host``). ``(None, None)``: the container
    sees no such path (no mount or workdir holds it), or the path is not absolute."""
    if not str(path).startswith("/"):
        return None, None
    path = str(path).rstrip("/")

    def under(path: str, base: str) -> bool:
        base = base.rstrip("/")
        return path == base or path.startswith(base + "/")

    if under(path, conf.workdir):
        return path, PROMPTS_WORKDIR
    if conf.runtime == "venv":  # the sampler reads the host's own paths
        if any(under(path, host) for host in node.mounts):
            return path, PROMPTS_MOUNT
        return path, PROMPTS_HOST
    seen = {**dict(node.mounts), **dict(conf.container.mounts if conf.container else {})}
    if conf.store:
        seen.setdefault(conf.store, conf.store)
    inside = sorted(((str(i).rstrip("/"), str(h).rstrip("/")) for h, i in seen.items() if under(path, str(i))),
                    key=lambda pair: -len(pair[0]))
    if not inside:
        return None, None
    mount, host = inside[0]
    return host + path[len(mount):], PROMPTS_MOUNT


def prompts_report(cfg: Mapping[str, Any], remote: str) -> tuple[list[str], list[str], list[str]]:
    """``pool --dry-run`` (:func:`yeto.launch.nc.cli.prompts_file_report`) of an ssh pool
    whose spec reads ``prompts.path`` ``remote``: ``(lines, problems, notes)``. With
    ``prompts_file`` the file goes to the host path behind ``remote`` on every node at each
    start (checked there by size and sha256): a path no node's sampler sees is a problem.
    Without it the sampler reads the file the node has: a note says where, a ``WARNING``
    line when the container sees no such path (no mount or workdir holds it)."""
    try:
        conf = parse_config(dict(cfg))
    except (ValueError, TypeError, KeyError, OSError):
        return [], [], []  # offline_problems reports it
    rows = [(node, *prompts_host_path(conf, node, remote)) for node in conf.nodes]
    unseen = [node.name for node, host, _where in rows if host is None]
    hosts = sorted({str(host) for _node, host, _where in rows if host is not None})
    seen_text = f"workdir {conf.workdir}, the inventory mounts or provider.container.mounts"
    if cfg.get("prompts_file"):
        if unseen:
            return [], [f"prompts_file: the sampler spec's prompts.path {remote} is under no mount or workdir of "
                        f"nodes {unseen} ({seen_text}): an upload there is not seen by the sampler; point "
                        "prompts.path under the pool's workdir or a mount"], []
        return [f"prompts_file: uploaded to {', '.join(hosts)} on every node of the pool"], [], []
    if unseen:
        return [f"WARNING: the sampler spec reads prompts.path {remote}, which the sampler container on nodes "
                f"{unseen} cannot read (no mount or workdir holds it: {seen_text}); set prompts_file to upload "
                "it (checked by size and sha256 at every start), or point prompts.path under the workdir or a "
                "mount"], [], []
    wheres = sorted({where for _node, _host, where in rows})
    place = {PROMPTS_WORKDIR: f"the pool's workdir {conf.workdir}", PROMPTS_MOUNT: "an inventory or pool mount",
             PROMPTS_HOST: "the node's own filesystem"}
    where = " / ".join(place[w] for w in wheres)
    return [], [], [f"the sampler spec reads prompts.path {remote} from the node ({where}; on the host "
                    f"{', '.join(hosts)}): put the file there before the run, or set prompts_file "
                    "(uploaded and checked by size and sha256 at every start)"]


def spec_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The pool's sampler spec file as written (``{}`` when it cannot be read: the
    pool's own checks report that)."""
    try:
        return json.loads(Path(os.path.expanduser(str(cfg["spec"]))).read_text())
    except (KeyError, OSError, ValueError, TypeError):
        return {}


def node_image(conf: SshPoolConf, node: Node, spec: Mapping[str, Any]) -> str:
    """The image ``node`` runs (container runtime): ``container.image`` unless it is
    ``auto``; then the node's inventory image, else the site image catalog's verified
    image for the spec's model on the pool's nodes, else the engine image catalog's choice
    for what ``spec`` asks (raises ``ImageIncompatible`` when none serves it)."""
    return str(node_image_choice(conf, node, spec).ref)


def _explicit(choice: Any) -> bool:
    from .site_images import SOURCE_EXPLICIT  # noqa: PLC0415

    return not isinstance(choice, Exception) and choice.source == SOURCE_EXPLICIT


def node_image_choice(conf: SshPoolConf, node: Node, spec: Mapping[str, Any]):  # -> site_images.Choice
    """:func:`node_image` with where it came from (``site_images.Choice``): the pool's or
    the inventory's image (explicit), the site image catalog, or the engine image catalog
    (the built-in default). A broken site catalog raises ``SiteImagesError``."""
    choices = pool_image_choices(conf, spec, only=node)
    choice = choices[node.name]
    if isinstance(choice, Exception):
        raise choice
    return choice


def pool_image_choices(conf: SshPoolConf, spec: Mapping[str, Any], *,
                       only: Node | None = None) -> dict[str, Any]:
    """Every node's image choice (``site_images.Choice``, or the exception that refuses
    the node), the pool's nodes at once: the site catalog is read once and matched against
    all of them (``auto`` nodes without an inventory image), so a spare that replaces a
    node starts the same site image. ``only``: just that node's entry (a node outside
    the pool is matched alone)."""
    from . import site_images  # noqa: PLC0415

    assert conf.container is not None
    nodes = list(conf.nodes)
    if only is not None and only.name not in {n.name for n in nodes}:
        nodes = [only]
    wanted = [only] if only is not None else nodes
    if conf.container.image and conf.container.image != AUTO_IMAGE:
        return {n.name: site_images.Choice(conf.container.image, site_images.SOURCE_EXPLICIT) for n in wanted}
    out: dict[str, Any] = {n.name: site_images.Choice(n.image, site_images.SOURCE_EXPLICIT) for n in wanted if n.image}
    auto = [n for n in nodes if not n.image]
    if not [n for n in wanted if not n.image]:
        return out
    needs = engine_images.spec_needs(spec)
    model, pref = needs.pop("model"), needs.pop("engine_pref")
    try:
        site = site_images.select(None, model, "sampler", [profile_of(n) for n in auto], engine=pref,
                                  path=conf.site_images, default=None)
    except site_images.SiteImagesError as error:
        return {**out, **{n.name: error for n in wanted if not n.image}}
    for node in wanted:
        if node.image:
            continue
        if site.source == site_images.SOURCE_SITE:
            out[node.name] = site
            continue
        try:
            ref = engine_images.resolve(profile_of(node), model, pref, **needs).image.pull_ref  # type: ignore[union-attr]
        except engine_images.ImageIncompatible as error:
            out[node.name] = error
            continue
        out[node.name] = site_images.Choice(ref, site_images.SOURCE_DEFAULT, None, site.path, site.why)
    return out


def image_problems(conf: SshPoolConf, spec: Mapping[str, Any]) -> list[str]:
    """``pool --dry-run``: every node's image serves the spec (a catalog image is held to
    the node's family, card architecture and HBM and to the spec's model, reload mode,
    precision and rollout fields; a site image is the site's)."""
    if conf.runtime != "container" or conf.container is None:
        return []
    needs = engine_images.spec_needs(spec)
    model, pref = needs.pop("model"), needs.pop("engine_pref")
    out = []
    from .site_images import SOURCE_EXPLICIT, SiteImagesError  # noqa: PLC0415

    choices = pool_image_choices(conf, spec)  # what the pool's first start fixes (SshSamplerPoolOps.image_for)
    for node in conf.nodes:
        try:
            choice = choices[node.name]
            if isinstance(choice, Exception):
                raise choice
            if choice.source != SOURCE_EXPLICIT:  # the dry-run names an image the pool chose itself
                print(f"ssh provider: node {node.name}: {choice.line('sampler')}")
            engine_images.resolve(profile_of(node), model, pref, image=str(choice.ref), **needs)
        except (engine_images.ImageIncompatible, SiteImagesError) as error:
            out.append(f"ssh provider: node {node.name}: {error}")
    return out


def egress_targets(cfg: Mapping[str, Any]) -> list[tuple[Node, tuple[Any, ...]]]:
    """``(node, endpoints)`` of each node of the pool: what its sampler reaches from the
    node's network (``yeto run up --dry-run`` checks it there over SSH): the run store's
    object API (the r2 data path; nothing for a node-local store or the syncer path) and
    the registry of the node's image (container runtime). The store endpoint is what the
    node's sampler takes (``yeto.node.store.STORE_URI_SHELL``): the ``r2_env_file``'s
    ``AWS_ENDPOINT_URL`` and nothing else, AWS S3 (in the file's region) when it is empty.
    Raises ValueError on a config the pool itself refuses."""
    from .nc.elastic import read_env_file  # noqa: PLC0415
    from .preflight.gate import registry_endpoint, s3_endpoint  # noqa: PLC0415

    conf = parse_config(dict(cfg))
    store = None
    if not conf.store and cfg.get("data_path", "r2") == "r2" and cfg.get("r2_env_file"):
        env = read_env_file(os.path.expanduser(str(cfg["r2_env_file"])))
        bucket = env.get(str(cfg.get("bucket_env") or "R2_BUCKET"))
        if bucket:
            region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
            store = s3_endpoint(env.get("AWS_ENDPOINT_URL"), region,
                                f"s3://{bucket}/runs/{str(cfg.get('run_prefix') or '').strip('/')}")
    spec = spec_of(cfg)
    out = []
    for node in conf.nodes:
        registry = None
        if conf.runtime == "container" and conf.container is not None:
            try:
                registry = registry_endpoint(conf.container.pull_ref or node_image(conf, node, spec))
            except engine_images.ImageIncompatible:
                registry = None  # the pool's own dry run refuses the node (image_problems)
        out.append((node, tuple(e for e in (store, registry) if e is not None)))
    return out


def required_pool_keys(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """The pool keys a ``kind: ssh`` pool needs (``r2_env_file`` only on the r2 data path)."""
    section = cfg.get("provider") or {}
    if section.get("store") or cfg.get("data_path", "r2") != "r2":
        return ("spec", "run_prefix")
    return ("spec", "run_prefix", "r2_env_file")
