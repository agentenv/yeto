"""Task environments on sandbox hosts: the ``agentic.envs`` spec section.

``agentic.envs`` (also a standalone JSON file for ``yeto sandbox --envs``)::

    {
      "run": "<label>",                  # containers yeto starts carry yeto.sandbox.run=<label>
      "ready_key": "sandbox/ready.json", # readiness manifest, relative to the run store
      "require_validation": true,        # a task is dispensable only after oracle/nop passed
      "affinity": "strict",              # strict: a sampler takes tasks ready on its own host only
                                         # prefer: own host first, then sidecar-free tasks ready elsewhere
      "pull_parallel": 4, "build_parallel": 2, "validate_parallel": 4,
      "hosts": [{"name": "h1", "docker_host": "ssh://user@h1", "ssh": "user@h1",
                 "max_sandboxes": 96, "port_range": [19100, 19999], "sync_to": "/data/tasks"}],
      "tasksets": [{"name": "set", "tasks_dir": "./tasks", "ids": ["..."],
                    "image_source": "auto",          # auto (pull, else build) | pull | build
                    "image_repo": "yeto-sbx-set",    # tag of images built from environment/
                    "agent_base": {"image": "yeto-sbx-pi-base:1", "context": "./pi-base"},
                    "bake_runtime": "pi",            # bake an agent runtime into every task image
                    "sidecars_file": "./sidecars.json"}],
      "assign": {"<sampler id>": "h1"}   # optional pins; see below
    }

Several hosts: with ``agentic.setup.sandbox.docker_host`` set to ``"envs"`` each
sampler drives one host of ``hosts``, picked by its shard index (:func:`assign_host`:
weighted round robin, ``max_sandboxes`` the weight, a replacement keeps its shard's
host). ``max_sandboxes`` caps the trial containers of every sampler on that host
together (the dispenser gate counts them all).

Tasks are Harbor task directories (``task.toml``, ``environment/Dockerfile``
and/or ``[environment] docker_image``, ``tests/``, ``solution/``). Sidecars
(long-lived per-task services on the host, e.g. a verification server) are
declared per task in ``<task>/yeto-sidecars.json`` or for the whole set in
``sidecars_file`` (``{"tasks": {"<task id>": {"sidecars": [...], "prefetch": [...]}}}``).
A sidecar: ``name``, ``image``, optional ``build`` (context, relative to the
declaring file), ``port`` (container port), ``host_port`` (fixed; else one is
allocated from the host's ``port_range``), ``container_name`` (adopt an
existing container of that name instead of starting one), ``env``, ``health``
(HTTP path). Text files of a task may name ``{{yeto.sidecar.<name>.url}}``,
``.host``, ``.port`` and ``{{yeto.host.bridge_ip}}``; ``prepare`` renders them per
host into the staged copy of the task (bridge address and port differ by host).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_READY_KEY = "sandbox/ready.json"
DEFAULT_PORT_RANGE = (19100, 19999)
AFFINITIES = ("strict", "prefer")
IMAGE_SOURCES = ("auto", "pull", "build")
SIDECAR_FILE = "yeto-sidecars.json"


class EnvsError(ValueError):
    """A bad ``agentic.envs`` section or task directory."""


@dataclass(frozen=True)
class HostConfig:
    name: str
    docker_host: str
    ssh: str | None = None
    max_sandboxes: int | None = None
    port_range: tuple[int, int] = DEFAULT_PORT_RANGE
    sync_to: str | None = None
    tunnel_listen_host: str | None = None  # the sampler's reverse-tunnel address on this host


@dataclass(frozen=True)
class Sidecar:
    name: str
    image: str
    port: int
    build: str | None = None
    host_port: int | None = None
    container_name: str | None = None
    env: tuple[tuple[str, str], ...] = ()
    health: str | None = None

    def spec_hash(self) -> str:
        body = json.dumps({"image": self.image, "port": self.port, "env": list(self.env),
                           "host_port": self.host_port}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class TaskEnv:
    task_id: str
    taskset: str
    dir: Path
    image: str  # what the trial runs (after baking)
    source_image: str | None  # environment.docker_image as declared
    build_context: Path | None  # environment/ with a Dockerfile
    sidecars: tuple[Sidecar, ...] = ()
    prefetch: tuple[str, ...] = ()
    bake_runtime: str | None = None

    @property
    def portable(self) -> bool:
        """No host-bound service: any host with the image can run it."""
        return not self.sidecars


@dataclass
class TasksetConfig:
    name: str
    tasks_dir: Path
    ids: list[str] | None = None
    image_source: str = "auto"
    image_repo: str | None = None
    agent_base: dict[str, Any] | None = None
    bake_runtime: str | None = None
    sidecars_file: Path | None = None


@dataclass
class EnvsConfig:
    run: str
    hosts: list[HostConfig]
    tasksets: list[TasksetConfig]
    ready_key: str = DEFAULT_READY_KEY
    require_validation: bool = True
    affinity: str = "strict"
    pull_parallel: int = 4
    build_parallel: int = 2
    validate_parallel: int = 4
    raw: dict[str, Any] = field(default_factory=dict)

    def host(self, name: str) -> HostConfig:
        for h in self.hosts:
            if h.name == name:
                return h
        raise EnvsError(f"no sandbox host {name!r} in agentic.envs.hosts")

    def host_of_docker(self, docker_host: str | None) -> HostConfig | None:
        want = _norm_docker(docker_host)
        for h in self.hosts:
            if want and _norm_docker(h.docker_host) == want:
                return h
        return None


def _norm_docker(value: str | None) -> str:
    return str(value or "").strip().rstrip("/")


def _resolve(base: Path | None, value: str) -> Path:
    p = Path(os.path.expanduser(str(value)))
    return p if p.is_absolute() or base is None else (base / p)


def parse_envs(section: Mapping[str, Any], *, base_dir: str | os.PathLike[str] | None = None,
               run: str | None = None) -> EnvsConfig:
    """``agentic.envs`` -> :class:`EnvsConfig` (module doc). Paths are relative to ``base_dir``."""
    if not isinstance(section, Mapping):
        raise EnvsError("agentic.envs must be an object")
    base = Path(base_dir) if base_dir is not None else None
    known = {"run", "ready_key", "require_validation", "affinity", "pull_parallel", "build_parallel",
             "validate_parallel", "hosts", "tasksets", "assign"}
    unknown = sorted(set(section) - known)
    if unknown:
        raise EnvsError(f"agentic.envs: unknown keys {unknown}")
    hosts = []
    for h in section.get("hosts") or []:
        try:
            pr = h.get("port_range") or DEFAULT_PORT_RANGE
            hosts.append(HostConfig(name=str(h["name"]), docker_host=str(h["docker_host"]),
                                    ssh=h.get("ssh"), max_sandboxes=None if h.get("max_sandboxes") is None
                                    else int(h["max_sandboxes"]),
                                    port_range=(int(pr[0]), int(pr[1])), sync_to=h.get("sync_to"),
                                    tunnel_listen_host=h.get("tunnel_listen_host")))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise EnvsError(f"agentic.envs.hosts: bad entry {h!r}: {exc}") from exc
    if len({h.name for h in hosts}) != len(hosts):
        raise EnvsError("agentic.envs.hosts: duplicate host names")
    sets = []
    for t in section.get("tasksets") or []:
        try:
            src = str(t.get("image_source", "auto"))
            if src not in IMAGE_SOURCES:
                raise EnvsError(f"image_source must be one of {IMAGE_SOURCES}")
            sets.append(TasksetConfig(
                name=str(t["name"]), tasks_dir=_resolve(base, t["tasks_dir"]),
                ids=None if t.get("ids") is None else [str(i) for i in t["ids"]], image_source=src,
                image_repo=t.get("image_repo"), agent_base=dict(t["agent_base"]) if t.get("agent_base") else None,
                bake_runtime=t.get("bake_runtime"),
                sidecars_file=_resolve(base, t["sidecars_file"]) if t.get("sidecars_file") else None))
        except (KeyError, TypeError) as exc:
            raise EnvsError(f"agentic.envs.tasksets: bad entry {t!r}: {exc}") from exc
    affinity = str(section.get("affinity", "strict"))
    assign = section.get("assign") or {}
    if not isinstance(assign, Mapping) or any(str(v) not in {h.name for h in hosts} for v in assign.values()):
        raise EnvsError("agentic.envs.assign must map sampler ids to host names of agentic.envs.hosts")
    if affinity not in AFFINITIES:
        raise EnvsError(f"agentic.envs.affinity must be one of {AFFINITIES}")
    label = str(run or section.get("run") or "").strip()
    if not label or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", label):
        raise EnvsError("agentic.envs.run (or --run) must be a label of [A-Za-z0-9_.-], up to 63 chars")
    return EnvsConfig(run=label, hosts=hosts, tasksets=sets,
                      ready_key=str(section.get("ready_key") or DEFAULT_READY_KEY),
                      require_validation=bool(section.get("require_validation", True)), affinity=affinity,
                      pull_parallel=max(1, int(section.get("pull_parallel", 4))),
                      build_parallel=max(1, int(section.get("build_parallel", 2))),
                      validate_parallel=max(1, int(section.get("validate_parallel", 4))), raw=dict(section))


def load_envs(path: str | os.PathLike[str], *, run: str | None = None) -> EnvsConfig:
    """A spec file (its ``agentic.envs``) or a standalone envs file."""
    p = Path(path).expanduser()
    doc = json.loads(p.read_text())
    section = (doc.get("agentic") or {}).get("envs") if isinstance(doc.get("agentic"), Mapping) else None
    return parse_envs(section if section is not None else doc, base_dir=p.parent, run=run)


# ------------------------------------------------------------------ tasks


def _sidecars(decl: Mapping[str, Any], base: Path) -> tuple[tuple[Sidecar, ...], tuple[str, ...]]:
    out = []
    for s in decl.get("sidecars") or []:
        try:
            out.append(Sidecar(name=str(s["name"]), image=str(s["image"]), port=int(s["port"]),
                               build=str(_resolve(base, s["build"])) if s.get("build") else None,
                               host_port=None if s.get("host_port") is None else int(s["host_port"]),
                               container_name=s.get("container_name"),
                               env=tuple(sorted((str(k), str(v)) for k, v in (s.get("env") or {}).items())),
                               health=s.get("health")))
        except (KeyError, TypeError, ValueError) as exc:
            raise EnvsError(f"bad sidecar {s!r}: {exc}") from exc
    return tuple(out), tuple(str(i) for i in decl.get("prefetch") or [])


def _tag_part(text: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", text.lower()).strip("-.")[:120] or "task"


def load_taskset(ts: TasksetConfig) -> list[TaskEnv]:
    root = ts.tasks_dir
    if not root.is_dir():
        raise EnvsError(f"taskset {ts.name}: no directory {root}")
    shared: Mapping[str, Any] = {}
    if ts.sidecars_file is not None:
        shared = (json.loads(ts.sidecars_file.read_text()).get("tasks") or {})
    names = ts.ids if ts.ids is not None else sorted(p.name for p in root.iterdir() if (p / "task.toml").is_file())
    tasks = []
    for tid in names:
        d = root / tid
        if not (d / "task.toml").is_file():
            raise EnvsError(f"taskset {ts.name}: {d} has no task.toml")
        meta = tomllib.loads((d / "task.toml").read_text())
        env = meta.get("environment") or {}
        declared = env.get("docker_image")
        context = d / "environment" if (d / "environment" / "Dockerfile").is_file() else None
        if not declared and context is None:
            raise EnvsError(f"task {tid}: neither environment.docker_image nor environment/Dockerfile")
        image = declared or f"{ts.image_repo or 'yeto-sbx-' + _tag_part(ts.name)}:{_tag_part(tid)}"
        sidecars: tuple[Sidecar, ...] = ()
        prefetch: tuple[str, ...] = ()
        if (d / SIDECAR_FILE).is_file():
            sidecars, prefetch = _sidecars(json.loads((d / SIDECAR_FILE).read_text()), d)
        elif tid in shared:
            sidecars, prefetch = _sidecars(shared[tid], ts.sidecars_file.parent)  # type: ignore[union-attr]
        bake = ts.bake_runtime
        final = image
        if bake:
            digest = hashlib.sha256(runtime_layer(bake).encode()).hexdigest()[:8]
            final = f"{ts.image_repo or 'yeto-sbx-' + _tag_part(ts.name)}:{_tag_part(tid)}-{bake}-{digest}"
        tasks.append(TaskEnv(task_id=tid, taskset=ts.name, dir=d, image=final, source_image=image,
                             build_context=context, sidecars=sidecars, prefetch=prefetch, bake_runtime=bake))
    return tasks


# ---------------------------------------------------------------- runtimes


def runtime_layer(name: str) -> str:
    """Dockerfile lines that install an agent runtime (``FROM`` excluded)."""
    if name == "pi":
        from yeto.rl.harness.harbor.pins import PI_PACKAGE, PI_VERSION  # noqa: PLC0415

        try:
            from harbor.agents.installed.node_install import nvm_node_install_snippet  # noqa: PLC0415

            node = nvm_node_install_snippet()
        except ImportError:  # the operator machine may not have Harbor
            node = ("curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && "
                    ". ~/.nvm/nvm.sh && nvm install 22")
        script = (f"set -e; {node} && . ~/.nvm/nvm.sh && npm install -g {PI_PACKAGE}@{PI_VERSION} && pi --version")
        return ("USER root\n"
                "RUN (command -v curl >/dev/null || (apt-get update && DEBIAN_FRONTEND=noninteractive "
                "apt-get install -y curl ca-certificates)) && rm -rf /var/lib/apt/lists/*\n"
                f"RUN bash -lc {json.dumps(script)}\n")
    raise EnvsError(f"unknown agent runtime {name!r} (known: pi)")


def all_tasks(cfg: EnvsConfig, ids: Sequence[str] | None = None) -> list[TaskEnv]:
    tasks = [t for ts in cfg.tasksets for t in load_taskset(ts)]
    if len({t.task_id for t in tasks}) != len(tasks):
        raise EnvsError("two tasksets declare the same task id")
    if ids:
        want = set(ids)
        tasks = [t for t in tasks if t.task_id in want]
    return tasks


# -- sampler -> sandbox host assignment ------------------------------------------------

ENVS_DOCKER_HOST = "envs"  # agentic.setup.sandbox.docker_host value: pick from agentic.envs.hosts


def shard_index(sampler_id: Any) -> int:
    """The trailing ``-<k>`` of a sampler id (``<run>-<k>``), the pool's shard number."""
    tail = str(sampler_id or "").rpartition("-")[2]
    if not tail.isdigit():
        raise EnvsError(f"sampler_id {sampler_id!r} does not end in -<shard index>: a sandbox host "
                        "cannot be assigned from agentic.envs.hosts")
    return int(tail)


def host_sequence(hosts: Sequence[HostConfig], n: int) -> list[HostConfig]:
    """The host of shards ``0 .. n-1``: smooth weighted round robin over ``hosts`` with
    ``max_sandboxes`` as the weight (equal weights when any host has none). Shard ``k``
    always gets the same host for the same host list, so a replacement sampler of a
    shard keeps its predecessor's host (its staged tasks and sidecars), and
    ``yeto run scale --add`` continues the sequence. Over any window of shards the
    share of each host follows its capacity."""
    if not hosts:
        raise EnvsError("agentic.envs.hosts is empty: no sandbox host to assign")
    weights = [h.max_sandboxes for h in hosts]
    if any(w is None or w <= 0 for w in weights):
        weights = [1] * len(hosts)
    total = sum(weights)
    current = [0] * len(hosts)
    out: list[HostConfig] = []
    for _ in range(max(0, int(n))):
        for i, w in enumerate(weights):
            current[i] += w
        best = max(range(len(hosts)), key=lambda i: (current[i], -i))
        current[best] -= total
        out.append(hosts[best])
    return out


def assign_host(cfg: EnvsConfig, sampler_id: Any) -> HostConfig:
    """The sandbox host of sampler ``sampler_id`` (:func:`host_sequence`); a pin in
    ``agentic.envs.assign`` (``{"<sampler id>": "<host name>"}``) wins."""
    pins = cfg.raw.get("assign") or {}
    if str(sampler_id) in pins:
        return cfg.host(str(pins[str(sampler_id)]))
    k = shard_index(sampler_id)
    return host_sequence(cfg.hosts, k + 1)[k]


def assignment_report(cfg: EnvsConfig, sampler_ids: Sequence[str], *, per_sampler: int | None) -> dict[str, Any]:
    """Which samplers drive which host and whether each host has room for them:
    ``per_sampler`` trial containers at once per sampler (its ``HARBOR_MAX_SANDBOXES``)
    against the host's ``max_sandboxes``. ``warnings``: hosts over capacity, without
    ``max_sandboxes`` (the gate then cannot hold claims back) or without a sampler."""
    rows: dict[str, dict[str, Any]] = {h.name: {"samplers": [], "max_sandboxes": h.max_sandboxes,
                                                 "docker_host": h.docker_host} for h in cfg.hosts}
    for sid in sampler_ids:
        rows[assign_host(cfg, sid).name]["samplers"].append(str(sid))
    warnings = []
    for name, row in rows.items():
        need = None if per_sampler is None else len(row["samplers"]) * int(per_sampler)
        row["containers_needed"] = need
        cap = row["max_sandboxes"]
        if cap is None:
            warnings.append(f"{name}: no max_sandboxes; the dispenser cannot hold claims back when the host is full")
        elif need is not None and need > cap:
            warnings.append(f"{name}: {len(row['samplers'])} samplers x {per_sampler} trials = {need} containers "
                            f"> max_sandboxes {cap}; add a host or lower HARBOR_MAX_SANDBOXES "
                            "(the samplers would wait for room: rl_dispense_wait_capacity)")
        if not row["samplers"]:
            warnings.append(f"{name}: no sampler is assigned to it")
    return {"hosts": rows, "warnings": warnings}
