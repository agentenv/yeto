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
  (``require_idle``: the compute apps of ``nvidia-smi`` / ``ppu-smi``, ``amd-smi
  process``, the process table of ``npu-smi info``), the host devices a rocm
  container needs (``/dev/kfd``, ``/dev/dri``), and the free space of ``workdir``
  (``min_free_gib``); a failed check refuses the start before anything runs;
* the yeto tree of the launcher's commit (``git archive``) under
  ``<workdir>/yeto/<sha>``, the member's rendered spec under
  ``<workdir>/sampler/<shard>.json``;
* the sampler, ``python -m yeto.rl.engine.engine_sampler --spec``, detached,
  either in a container (``runtime: container``: podman or docker) or as a
  process of a venv on the host (``runtime: venv``: no container runtime);
* its log in ``<workdir>/logs/<shard>.log`` (container: ``<engine> logs``),
  read back by :meth:`SshSamplerPoolOps.logs`; its event tape next to the spec.

Stop is by PID, never by name matching: the start script writes the PID of the
sampler process before it ``exec``s python; ``stop`` checks with ``ps -o args=``
that this PID is still the engine sampler, sends it SIGTERM (the sampler runs
with ``YETO_SPOT_RECLAIM_MARKER`` set, so SIGTERM is its orderly leave: it stops
the engine and exits), waits ``grace_s``, and only then kills what is left of
the process group of that start. A container is stopped with ``<engine> stop -t
grace_s`` (SIGTERM to its PID 1, the sampler).

Device defaults by family (:class:`~yeto.launch.node_inventory.NodeProfile`, an
inventory ``docker_args`` wins): cuda ``--gpus`` (docker) or CDI
``nvidia.com/gpu=<i>`` (podman) and ``CUDA_VISIBLE_DEVICES`` as the container numbers
the cards; rocm ``/dev/kfd``, ``/dev/dri``, the video group and seccomp off, with
``HIP_VISIBLE_DEVICES``; ascend ``/dev/davinci<i>`` plus the host devices and driver
mounts the preflight found, with ``ASCEND_RT_VISIBLE_DEVICES`` when ``devices`` is
set. Every sampler gets ``YETO_DEVICE_FAMILY`` (torch reads ROCm as ``cuda``).

``container.image: "auto"`` takes each node's image from the engine image catalog
(:func:`yeto.launch.engine_images.resolve` with what the sampler spec asks: model,
engine, reload mode, precision, rollout fields), so a pool of mixed rocm and cuda
nodes runs one spec; ``pool --dry-run`` refuses a node no image serves.

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
from .pool.provider import ProviderCaps
from .pool.status import LeaseStatus

KIND = "ssh"
REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "yeto-ssh-sampler"  # prefix of a start's marker (start_marker): run, shard, ticket
MARK_ENV = "YETO_SSH_SAMPLER_MARK"  # the sampler process carries its start's marker in this variable
RUNTIMES = ("container", "venv")
ENGINES = ("podman", "docker")
PULL_POLICIES = ("missing", "always", "never")
CONFIG_KEYS = frozenset({"kind", "inventory", "nodes", "workdir", "runtime", "container", "venv", "env",
                         "devices", "store", "preflight", "grace_s", "price_usd_h", "start_timeout_s"})
CONTAINER_KEYS = frozenset({"engine", "image", "pull_ref", "pull", "security_opt", "sudo", "args", "python",
                            "shm_size", "mounts"})
VENV_KEYS = frozenset({"python", "source", "ld_preload", "pythonpath"})
PREFLIGHT_KEYS = frozenset({"min_free_gib", "require_idle", "min_cards"})
AUTO_IMAGE = "auto"  # container.image: each node's image from the engine image catalog
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
    require_idle: bool
    min_cards: int | None
    grace_s: int
    price_usd_h: float | None


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
    return SshPoolConf(nodes=tuple(nodes), workdir=workdir.rstrip("/"), runtime=runtime, container=container,
                       venv=venv, env={str(k): str(v) for k, v in env.items()}, devices=devices,
                       store=str(store) if store else None, min_free_gib=float(pre.get("min_free_gib", 20)),
                       require_idle=bool(pre.get("require_idle", True)),
                       min_cards=int(pre["min_cards"]) if pre.get("min_cards") is not None else None,
                       grace_s=int(section.get("grace_s", 120)),
                       price_usd_h=float(price) if price is not None else None)


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
    idle = idle_command(node) if conf.require_idle else None
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
    if conf.require_idle:
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
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 2 and cells[1].isdigit() and cells[0].lower() in index_of:
            busy.add(index_of[cells[0].lower()])
    return sorted(busy)


def busy_amd(text: str) -> list[int] | None:
    """Card indexes with a process in ``amd-smi process --json`` (a list of ``{"gpu": i,
    "process_list": [...]}``; an entry is ``{"process_info": {..."pid"...}}`` up to ROCm
    6.3 and a flat ``{"pid": ...}`` later; "No running processes detected" is idle).
    None when the output is not that JSON (amd-smi missing or failing)."""
    start = text.find("[")
    if start < 0:
        start = text.find("{")
    try:
        doc = json.loads(text[start:]) if start >= 0 else None
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
                 env: Mapping[str, str], env_file: str | None) -> str:
    """The venv start script: env scripts, environment, PID record, ``exec`` python."""
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
    lines += [f"cd {shlex.quote(conf.workdir)}", f"echo $$ > {shlex.quote(pid_path)}",
              f"exec {shlex.quote(v.python)} -m yeto.rl.engine.engine_sampler --spec {shlex.quote(spec_path)}"]
    return "\n".join(lines) + "\n"


_STORE_URI = ('export YETO_RL_STORE_S3_URI="s3://${R2_BUCKET}/runs/${YETO_NC_RUN_PREFIX}" '
              'YETO_RL_STORE_S3_ENDPOINT_URL="$AWS_ENDPOINT_URL"')


def venv_launch_command(p: Mapping[str, str], marker: str) -> str:
    """Detach the start script in its own session (``$0``: the start's marker); the
    session's PGID and the marker are recorded, the script records the sampler's own
    PID. The log starts empty: a line of an earlier run is never read as this one's."""
    inner = f"bash {shlex.quote(p['script'])} >> {shlex.quote(p['log'])} 2>&1; echo $? > {shlex.quote(p['exit'])}"
    return (f"rm -f {p['exit']} {p['pid']}; mkdir -p {p['logs']}; : > {p['log']}; "
            f"printf '%s\\n' {shlex.quote(marker)} > {p['marker']}; "
            f"{{ setsid bash -c {shlex.quote(inner)} {shlex.quote(marker)} </dev/null >/dev/null 2>&1 & "
            f"echo $! > {p['pgid']}; }}; "
            f"for i in 1 2 3 4 5 6 7 8 9 10; do {_alive(p)} && break; sleep 1; done; "
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
                       host_paths: Sequence[str] | None = None) -> list[str]:
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
    # what the preflight found on the host is passed (a missing -v source fails the run)
    argv += list(pool_profile(node, conf).devices(cards, engine=c.engine, host_paths=host_paths).args)
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
    body = f'export PYTHONPATH={shlex.quote(yeto_dir)}"${{PYTHONPATH:+:$PYTHONPATH}}"; ' + \
        ((_STORE_URI + "; ") if env_file else "") + \
        f"exec {shlex.quote(c.python)} -m yeto.rl.engine.engine_sampler --spec {shlex.quote(spec_path)}"
    return [*argv, "--entrypoint", "bash", image, "-c", body]


# -- the provider ---------------------------------------------------------------


class SshNodeProvider:
    """Slots are the configured nodes; tickets live in a local record. Never orders,
    never powers anything off: ``release`` only marks the ticket ended (the ops stop
    the sampler first)."""

    kind = KIND
    caps = ProviderCaps(lease_bounded=False, notice_s=0.0, currency="usd", cold_start_s=300)

    def __init__(self, run_id: str, conf: SshPoolConf, *, slots: Sequence[int], state_path: Path):
        self.run_id, self.conf, self.slots, self.state_path = run_id, conf, list(slots), state_path
        self._lock = threading.Lock()
        bad = [s for s in self.slots if not 0 <= s < len(conf.nodes)]
        if bad:
            raise ValueError(f"ssh provider: slots {bad} have no node (nodes: {[n.name for n in conf.nodes]})")

    def _rows(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, rows: dict[str, dict[str, Any]]) -> None:
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
                            "ordered_at": time.time()}
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
                rows[ticket] = dict(rows[ticket], state="released", released_at=time.time())
                self._save(rows)
        return {"ticket": ticket, "state": "released"}

    def released_by_us(self, ticket: str) -> bool:
        return self._rows().get(ticket, {}).get("state") == "released"

    def node(self, account: int, ticket: str) -> Node:  # noqa: ARG002
        return self.node_of_ticket(ticket)

    def balance(self) -> int | None:
        return None

    def account_free(self, account: int) -> bool:
        return not any(int(r["slot"]) == int(account) and r["state"] == "active" for r in self._rows().values())

    def run_tickets(self) -> list[tuple[int, str]]:
        return [(int(r["slot"]), t) for t, r in self._rows().items() if r["state"] == "active"]

    def hourly_cost(self, ticket: str | None = None) -> float | None:  # noqa: ARG002
        return self.conf.price_usd_h


# -- the pool ops ---------------------------------------------------------------


class SshSamplerPoolOps:
    """The pool's ``PoolOps`` for samplers on the operator's own SSH nodes."""

    start_in_background = True  # an engine cold start on an NPU takes minutes

    def __init__(self, cfg: dict[str, Any], *, conf: SshPoolConf | None = None,
                 run: Callable[..., Any] = subprocess.run, state_dir: Path | None = None,
                 code_ref: str = "HEAD"):
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
                                        state_path=self.state_dir / f"ssh-pool-{self.run_id}.json")
        self._shard_of: dict[str, str] = {}

    # -- shell on a node
    def ssh(self, node: Node, command: str, *, timeout: float = 600, input: str | bytes | None = None,
            stdin: Any = None) -> Any:
        route = [*node.ssh_options(), node.host]
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
        env = {"PYTHONUNBUFFERED": "1",
               # SIGTERM is the orderly leave (sampler_resume.install_marker_signals)
               "YETO_SPOT_RECLAIM_MARKER": f"{self.conf.workdir}/sampler/{m.shard}.reclaim"}
        profile = pool_profile(node, self.conf)
        if profile.family != "ascend" or self.conf.devices is not None:  # ascend: only an explicit card list
            env.update(profile.devices(self.cards(node)).env)
        env.update(profile.family_env())
        if self._uses_r2():
            prefix = str(self.cfg["run_prefix"]).strip("/")
            env.update({"YETO_RL_STORE_IMPL": "yeto.cloud.s3_store:store_from_env",
                        "YETO_RL_STORE_S3_STAGE": f"{self.conf.workdir}/stage", "YETO_RL_STORE_S3_REGION": "auto",
                        "YETO_RL_STORE_S3_CREDENTIALS": "aws-env", "YETO_NC_RUN_PREFIX": prefix,
                        "YETO_RL_RELEASE_SPREAD_KEY": f"{prefix}/{m.shard}"})
        env.update(self.conf.env)
        return env

    def _uses_r2(self) -> bool:
        return not self.conf.store and self.cfg.get("data_path", "r2") == "r2"

    def start(self, m, shards: list[str]) -> None:
        node = self.provider.node_of_ticket(m.ticket)
        self._shard_of[m.ticket] = m.shard
        p = node_paths(self.conf.workdir, m.shard)
        facts = self.preflight(node)
        _log(f"ssh pool: {node.name} preflight ok {facts}")
        self._stop_on(node, m.shard)  # a restarted pool: never two samplers of one shard
        yeto_dir = self.upload_code(node)
        self.put(node, json.dumps(self.spec_for(m, shards), indent=1), p["spec"])
        env_file = None
        if self._uses_r2():
            env_file = f"{self.conf.workdir}/sampler/.store.env"
            self.put(node, Path(os.path.expanduser(str(self.cfg["r2_env_file"]))).read_text(), env_file, mode="600")
        env = self.sampler_env(node, m)
        if self.conf.runtime == "venv":
            marker = start_marker(self.run_id, m.shard, m.ticket)
            env[MARK_ENV] = marker
            self.put(node, start_script(self.conf, yeto_dir=yeto_dir, spec_path=p["spec"], pid_path=p["pid"],
                                        env=env, env_file=env_file), p["script"], mode="700")
            out = self.ssh(node, venv_launch_command(p, marker), timeout=120)
        else:
            image = self.image_for(node)
            name = container_name(m.shard)
            argv = container_run_argv(self.conf, node, name=name, image=str(image), yeto_dir=yeto_dir,
                                      spec_path=p["spec"], env=env, env_file=env_file, cards=self.cards(node),
                                      host_paths=facts.get("host_paths"))
            out = self.ssh(node, f"{container_pull_command(self.conf, str(image))} && "
                                 f"{{ {_ce(self.conf)} rm -f {name} >/dev/null 2>&1; true; }} && "
                                 f"{shlex.join(argv)} && echo YETO_SSH_STARTED container={name}", timeout=3600)
        text = _text(out.stdout)
        if out.returncode != 0 or "YETO_SSH_STARTED" not in text:
            raise SshSamplerError(f"sampler {m.shard} on {node.name} did not start (exit {out.returncode}): "
                                  f"{(text + _text(out.stderr))[-800:]}")
        _log(f"ssh pool: {m.shard} on {node.name}: {text.strip().splitlines()[-1]}")

    def image_for(self, node: Node) -> str:
        """The image a node runs: ``container.image`` (``auto``: the node's inventory
        image, else the catalog's for what the spec asks), else the node's image."""
        return node_image(self.conf, node, spec_of(self.cfg))

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
        grace = int(self.conf.grace_s if grace_s is None else grace_s)
        p = node_paths(self.conf.workdir, shard)
        if self.conf.runtime == "venv":
            cmd = f"{venv_stop_command(p, grace)}; tail -n 300 {p['log']} > {p['logs']}/{shard}.last.log 2>/dev/null"
        else:
            cmd = container_stop_command(self.conf, p, container_name(shard), grace)
        out = self.ssh(node, cmd, timeout=grace + 180)
        return _text(out.stdout)

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
        from .nc import cli, elastic  # noqa: PLC0415

        env = elastic.read_env_file(self.cfg["r2_env_file"])
        bucket = env.get(self.cfg.get("bucket_env", "R2_BUCKET"))
        if not bucket:
            raise cli.NcError(f"no {self.cfg.get('bucket_env', 'R2_BUCKET')} in the store env file")
        return elastic.s3_fence_store(f"s3://{bucket}/runs/{self.cfg['run_prefix'].strip('/')}",
                                      self.cfg["r2_env_file"])

    def fence(self, m) -> Any:
        if self.conf.store:
            return {"fenced_by_stop": self.stop(m)}
        return self.data_path().fence_shard(m.shard, by=m.ticket)

    def fence_incarnation(self, m) -> Any:
        if self.conf.store:
            return {"fenced_by_stop": self.stop(m)}
        return self.data_path().fence_incarnation(m.shard, by=m.ticket)

    def drain(self, m) -> None:
        if m.shard:
            try:
                _log(f"ssh pool: stop {m.shard}: {self.stop(m).strip()[-300:]}")
            except subprocess.TimeoutExpired:
                _log(f"ssh pool: stop {m.shard} timed out; the ticket is ended, check the node")
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
    problems += image_problems(conf, spec_of(cfg))
    return problems


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
