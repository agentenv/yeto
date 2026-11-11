"""Agentic sampler node setup, declared in the sampler spec (``agentic.setup``).

An agentic sampler runs Harbor trials in-process: Harbor and the docker CLI must
be importable / on ``PATH`` inside the sampler container, the docker CLI must
reach the sandbox host's daemon over SSH, the sandboxes must reach the
sampler's records proxy through a reverse tunnel, and the task directories and
the proxy's path secret must be on the node. This module makes a node so, from
the spec section below, as one idempotent step that runs before the sampler
container starts (``ssh-sampler start``; every route that starts a sampler
through :func:`yeto.launch.nc.ssh_sampler.start_sampler`, so a replacement
node of a pool gets it too). A second run on a ready node changes nothing.

The sandbox host itself (accounts, images, the pi build inside the images) is
prepared on the sandbox host side, not here.

Spec section (container paths; every key optional unless marked)::

    "agentic": {
      "proxy": {"host": "127.0.0.1", "port": 18080},
      "setup": {
        "root": "/work/harbor",
        "harbor": {"version": "0.20.0", "keep": ["httpx", "filelock"]},
        "docker_cli": {"version": "29.0.2", "compose_version": "2.40.3"},
        "sandbox": {"docker_host": "ssh://<user>@<host>[:<port>]" | "envs", # required
                    "alias": "yeto-sandbox",
                    "key": {"path": "<local file>"} | {"env": "<VAR>"},      # required
                    "known_hosts": {"path": ...} | {"env": ...}},
        "tunnel": {"listen": "<private ip>:<port>",                          # required for a tunnel
                   "user": ..., "host": ..., "port": 22,
                   "key": {"path": ...} | {"env": ...},
                   "target_port": <agentic.proxy.port>,
                   "settle_s": 5, "retry_s": 2, "max_backoff_s": 60, "ready_timeout_s": 90},
        "tasks": {"source": "<local dir>" | "store:<prefix>", "ids": [...] | "ids_file": "<local file>"},
        "records_secret": {"path": ...} | {"env": ...} | "generate"
      }
    }

Steps (each with a health check; a failure raises :class:`SetupError` naming the
step and what to do):

``layout``       the directories under ``root`` (owned by root: the container user)
``docker_cli``   static docker CLI + compose plugin of the pinned versions
``harbor``       ``pip --target <root>/site`` of the pinned Harbor build with the
                 container's Python; packages the image already has are pruned
                 from the target except ``keep``; checked by importing Harbor and
                 vLLM together and reading Harbor's version and commit
``sandbox``      the docker key, ssh config (``ControlMaster``) and known_hosts
                 under ``<root>/home/.ssh``; checked by ``docker version`` on the
                 sandbox daemon from the sampler image
``records_secret``  ``records-proxy.secret`` in ``<root>``'s home dir (0600, >= 24 chars)
``tasks``        the task directories under ``<root>/tasks`` (local tree or store
                 prefix); checked by a ``task.toml`` per wanted task
``tunnel``       a helper container (``--restart unless-stopped``) running
                 :mod:`.agentic_node_agent` ``tunnel``: ``ssh -N -R`` with the
                 tunnel key, which lives outside ``/work`` and so is never visible
                 to the sampler container; reconnects on every drop and writes
                 ``rl_agentic_tunnel`` up/down events to ``<root>/events/tunnel.jsonl``

Several sandbox hosts: ``sandbox.docker_host: "envs"`` takes the sampler's host
from ``agentic.envs.hosts`` (:func:`yeto.sandbox.envs.assign_host`: by shard index,
weighted by ``max_sandboxes``; a replacement of a shard keeps the shard's host). The
docker connection and the reverse tunnel then both go to that host (``tunnel.host``
must be unset); a host entry's ``tunnel_listen_host`` replaces the address part of
``tunnel.listen`` on that host. The sampler's env names the host
(``YETO_SANDBOX_HOST``) so the dispenser gate counts that host's containers.

The sampler container then gets :meth:`AgenticSetup.container_env` (``HOME``,
``PATH``, ``DOCKER_HOST``, ``HARBOR_*`` directories, the proxy's public base and
secret file) as an env file ahead of the operator's ``--agentic-env`` file, which
still wins on every key it sets.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from yeto.rl.harness.harbor.pins import HARBOR_COMMIT, HARBOR_REPO, HARBOR_VERSION

SETUP_EVENT = "rl_agentic_setup"
TUNNEL_EVENT = "rl_agentic_tunnel"
DEFAULT_ROOT = "/work/harbor"
DEFAULT_ALIAS = "yeto-sandbox"
ENVS_DOCKER_HOST = "envs"  # yeto.sandbox.envs.ENVS_DOCKER_HOST: the host comes from agentic.envs.hosts
TUNNEL_ALIAS = "yeto-tunnel"
TUNNEL_CONTAINER = "yeto-agentic-tunnel"
DOCKER_CLI_VERSION = "29.0.2"
COMPOSE_VERSION = "2.40.3"
HARBOR_KEEP = ("httpx", "filelock")  # moves the MI355X vLLM image's Harbor check accepted
MIN_SECRET_CHARS = 24
DOCKER_STATIC_URL = "https://download.docker.com/linux/static/stable/x86_64/docker-{v}.tgz"
COMPOSE_URL = "https://github.com/docker/compose/releases/download/v{v}/docker-compose-linux-x86_64"
STORE_PREFIX = "store:"
AWS_IMAGE = "amazon/aws-cli:latest"

_KEYS = {
    "": {"root", "harbor", "docker_cli", "sandbox", "tunnel", "tasks", "records_secret"},
    "harbor": {"version", "commit", "keep"},
    "docker_cli": {"version", "compose_version"},
    "sandbox": {"docker_host", "alias", "key", "known_hosts"},
    "tunnel": {"listen", "user", "host", "port", "key", "target_port", "settle_s", "retry_s", "max_backoff_s",
               "ready_timeout_s"},
    "tasks": {"source", "ids", "ids_file"},
}
_HANDED_OFF = {"images", "prefetch", "pi", "pi_cache"}  # the sandbox host side (agentic.envs)
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class SetupError(RuntimeError):
    """A setup step failed; ``step`` names it, the message says what to do."""

    def __init__(self, step: str, message: str):
        super().__init__(f"agentic setup [{step}]: {message}")
        self.step = step


# -- secrets -------------------------------------------------------------------

@dataclass(frozen=True)
class SecretRef:
    """A local file (``path``) or an environment variable of the launcher (``env``)."""

    path: str | None = None
    env: str | None = None

    @classmethod
    def parse(cls, value: Any, what: str) -> SecretRef:
        if isinstance(value, Mapping) and set(value) == {"path"} and value["path"]:
            return cls(path=str(value["path"]))
        if isinstance(value, Mapping) and set(value) == {"env"} and _NAME.match(str(value["env"] or "")):
            return cls(env=str(value["env"]))
        raise ValueError(f"{what} must be {{\"path\": <local file>}} or {{\"env\": <VAR>}}, got {value!r}")

    def describe(self) -> str:
        return f"file {self.path}" if self.path else f"env ${self.env}"

    def problem(self, environ: Mapping[str, str], *, private: bool) -> str | None:
        if self.env:
            return None if environ.get(self.env) else f"environment variable {self.env} is unset or empty"
        p = Path(self.path).expanduser()
        if not p.is_file():
            return f"{self.path} does not exist"
        if private and p.stat().st_mode & 0o077:
            return f"{self.path} must be mode 0600 (chmod 600)"
        return None

    def read(self, environ: Mapping[str, str]) -> str:
        if self.env:
            value = environ.get(self.env, "")
            if not value:
                raise ValueError(f"environment variable {self.env} is unset or empty")
            return value if value.endswith("\n") else value + "\n"
        return Path(self.path).expanduser().read_text()


# -- the spec section ------------------------------------------------------------

@dataclass(frozen=True)
class SandboxAccess:
    user: str
    host: str
    port: int
    alias: str
    key: SecretRef
    known_hosts: SecretRef | None


@dataclass(frozen=True)
class TunnelSpec:
    listen_host: str
    listen_port: int
    target_port: int
    user: str
    host: str
    port: int
    key: SecretRef
    settle_s: float
    retry_s: float
    max_backoff_s: float
    ready_timeout_s: float

    @property
    def listen(self) -> str:
        return f"{self.listen_host}:{self.listen_port}"


@dataclass(frozen=True)
class TasksSpec:
    source: str
    ids: tuple[str, ...]

    @property
    def store_prefix(self) -> str | None:
        return self.source[len(STORE_PREFIX):].strip("/") if self.source.startswith(STORE_PREFIX) else None


@dataclass(frozen=True)
class AgenticSetup:
    root: str
    harbor_version: str
    harbor_commit: str
    harbor_keep: tuple[str, ...]
    docker_cli_version: str
    compose_version: str
    sandbox: SandboxAccess
    tunnel: TunnelSpec | None
    tasks: TasksSpec | None
    records_secret: SecretRef | str | None
    notes: tuple[str, ...] = field(default=())
    sandbox_host: str | None = None  # the agentic.envs.hosts name of the sandbox host

    # -- container paths
    @property
    def home(self) -> str:
        return f"{self.root}/home"

    @property
    def site(self) -> str:
        return f"{self.root}/site"

    @property
    def secret_file(self) -> str:
        return f"{self.home}/records-proxy.secret"

    @property
    def tunnel_events(self) -> str:
        return f"{self.root}/events/tunnel.jsonl"

    def container_env(self, *, python_bin: str = "/opt/runtime-venv/bin") -> dict[str, str]:
        """The sampler container's environment for this setup (an env file ahead of
        the operator's agentic env file). ``PYTHONPATH`` is :meth:`python_path`."""
        env = {"HOME": self.home, "DOCKER_HOST": f"ssh://{self.sandbox.alias}",
               "PATH": f"{self.root}/bin:{python_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
               "HARBOR_ENV_TYPE": "remote-docker", "HARBOR_TASKS_DIR": f"{self.root}/tasks",
               "HARBOR_TRIALS_DIR": f"{self.root}/trials"}
        if self.records_secret is not None:
            env["YETO_RECORDS_PROXY_SECRET_FILE"] = self.secret_file
        if self.tunnel is not None:
            env["YETO_RECORDS_PROXY_PUBLIC_BASE"] = f"http://{self.tunnel.listen}"
        if self.sandbox_host is not None:
            env["YETO_SANDBOX_HOST"] = self.sandbox_host  # yeto.sandbox.gate.HOST_ENV
        return env

    def python_path(self, base: str = "/yeto") -> str:
        return f"{base}:{self.site}"

    def fingerprint(self) -> str:
        """A digest of the tunnel's settings (the helper container is replaced when it changes)."""
        t = self.tunnel
        doc = None if t is None else [t.listen, t.target_port, t.user, t.host, t.port, t.key.describe(),
                                      t.settle_s, t.retry_s, t.max_backoff_s]
        return hashlib.sha256(json.dumps(doc).encode()).hexdigest()[:16]


def _unknown(section: Mapping[str, Any], where: str) -> None:
    allowed = _KEYS[where]
    unknown = set(section) - allowed
    if unknown & _HANDED_OFF:
        raise ValueError(f"agentic.setup{'.' + where if where else ''}: {sorted(unknown & _HANDED_OFF)} are "
                         "sandbox host settings (image pre-pulls, the pi build): declare them in agentic.envs")
    if unknown:
        raise ValueError(f"agentic.setup{'.' + where if where else ''}: unknown keys {sorted(unknown)}; "
                         f"known: {sorted(allowed)}")


def _section(doc: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = doc.get(key) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"agentic.setup.{key} must be an object")
    _unknown(value, key)
    return dict(value)


def _shard_port(port_range: str, sampler_id: Any) -> int:
    """``tunnel.listen`` ``<ip>:<lo>-<hi>``: islands run at once and each holds its own
    reverse forward, so shard ``<run>-<k>`` listens on ``lo + k`` (a replacement node of
    the same shard takes the same port once the fenced node is gone). The tunnel key
    needs a permitlisten for every port of the range."""
    lo_s, _, hi_s = port_range.partition("-")
    lo = _port(lo_s, "agentic.setup.tunnel.listen port range start")
    hi = _port(hi_s, "agentic.setup.tunnel.listen port range end")
    if hi < lo:
        raise ValueError(f"agentic.setup.tunnel.listen port range {port_range!r} is empty")
    tail = str(sampler_id or "").rpartition("-")[2]
    if not tail.isdigit():
        raise ValueError(f"agentic.setup.tunnel.listen is a port range but the spec's sampler_id "
                         f"{sampler_id!r} does not end in -<shard index>")
    port = lo + int(tail)
    if port > hi:
        raise ValueError(f"agentic.setup.tunnel.listen range {port_range!r} has no port for shard "
                         f"{sampler_id!r}: widen it to at least one port per shard")
    return port


def _port(value: Any, what: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a port number, got {value!r}") from None
    if not 0 < port < 65536:
        raise ValueError(f"{what} must be in 1..65535, got {port}")
    return port


def parse(spec: Mapping[str, Any], *, base_dir: str | Path | None = None) -> AgenticSetup | None:
    """The ``agentic.setup`` section of ``spec`` (``None`` when absent). Raises
    ValueError on unknown keys and bad values; local files are not read
    (:func:`local_problems`). Relative local paths resolve against ``base_dir``."""
    agentic = spec.get("agentic") or {}
    doc = agentic.get("setup") if isinstance(agentic, Mapping) else None
    if doc is None:
        return None
    if not isinstance(doc, Mapping):
        raise ValueError("agentic.setup must be an object")
    _unknown(doc, "")
    notes: list[str] = []

    def local(path: str) -> str:
        p = Path(path).expanduser()
        return str(p if p.is_absolute() or base_dir is None else Path(base_dir) / p)

    def ref(value: Any, what: str) -> SecretRef:
        r = SecretRef.parse(value, what)
        return SecretRef(path=local(r.path)) if r.path else r

    root = str(doc.get("root", DEFAULT_ROOT)).rstrip("/")
    if not root.startswith("/work/") or ".." in root.split("/"):
        raise ValueError(f"agentic.setup.root {root!r} must be a directory under /work (the node's work mount)")

    harbor = _section(doc, "harbor")
    version = str(harbor.get("version", HARBOR_VERSION))
    commit = str(harbor.get("commit", HARBOR_COMMIT))
    if version != HARBOR_VERSION or commit != HARBOR_COMMIT:
        raise ValueError(f"agentic.setup.harbor: yeto is verified against Harbor {HARBOR_VERSION} at "
                         f"{HARBOR_COMMIT[:12]} (yeto/rl/harness/harbor/pins.py), got {version} at {commit[:12]}")
    keep = tuple(str(k).lower() for k in harbor.get("keep", HARBOR_KEEP))

    cli = _section(doc, "docker_cli")
    cli_version = str(cli.get("version", DOCKER_CLI_VERSION))
    compose = str(cli.get("compose_version", COMPOSE_VERSION)).lstrip("v")
    for what, v in (("docker_cli.version", cli_version), ("docker_cli.compose_version", compose)):
        if not _VERSION.match(v):
            raise ValueError(f"agentic.setup.{what} must be a pinned x.y.z version, got {v!r}")

    sb = _section(doc, "sandbox")
    envs_host = None
    if str(sb.get("docker_host") or "") == ENVS_DOCKER_HOST:
        envs_host = _envs_host(spec, agentic)
        url = urlsplit(envs_host.docker_host)
    else:
        url = urlsplit(str(sb.get("docker_host") or ""))
        envs_host = _envs_host_of(spec, agentic, url)
    if url.scheme != "ssh" or not url.hostname or not url.username:
        raise ValueError("agentic.setup.sandbox.docker_host must be ssh://<user>@<host>[:<port>], got "
                         f"{sb.get('docker_host')!r}")
    if "key" not in sb:
        raise ValueError("agentic.setup.sandbox.key is required (the docker key: {\"path\"} or {\"env\"})")
    alias = str(sb.get("alias", DEFAULT_ALIAS))
    if not _NAME.match(alias) or alias == TUNNEL_ALIAS:
        raise ValueError(f"agentic.setup.sandbox.alias {alias!r} must be a plain host alias")
    known = ref(sb["known_hosts"], "agentic.setup.sandbox.known_hosts") if sb.get("known_hosts") else None
    if known is None:
        notes.append("sandbox.known_hosts is unset: the first connection trusts the host key it sees "
                     "(StrictHostKeyChecking accept-new); pin it for production")
    sandbox = SandboxAccess(user=url.username, host=url.hostname, port=url.port or 22, alias=alias,
                            key=ref(sb["key"], "agentic.setup.sandbox.key"), known_hosts=known)

    tunnel = None
    tn = _section(doc, "tunnel")
    if tn:
        listen = str(tn.get("listen") or "")
        host, _, port = listen.rpartition(":")
        if not host or not port:
            raise ValueError(f"agentic.setup.tunnel.listen must be <private ip>:<port>, got {listen!r}")
        if envs_host is not None and str(sb.get("docker_host")) == ENVS_DOCKER_HOST:
            if tn.get("host"):
                raise ValueError("agentic.setup.tunnel.host must be unset with sandbox.docker_host \"envs\": "
                                 "the tunnel goes to the sampler's assigned sandbox host")
            host = envs_host.tunnel_listen_host or host
        proxy_port = (agentic.get("proxy") or {}).get("port")
        target = tn.get("target_port", proxy_port)
        if not target:
            raise ValueError("agentic.setup.tunnel needs a fixed records proxy port: set agentic.proxy.port "
                             "(or tunnel.target_port)")
        if proxy_port and int(target) != int(proxy_port):
            raise ValueError(f"agentic.setup.tunnel.target_port {target} is not agentic.proxy.port {proxy_port}")
        key = ref(tn["key"], "agentic.setup.tunnel.key") if tn.get("key") else sandbox.key
        if not tn.get("key"):
            notes.append("tunnel.key is unset: the tunnel uses the docker key; the guide (Sandbox host, SSH keys) recommends "
                         "two keys (the tunnel key never enters the sampler container)")
        if "-" in port:  # a port range: one listen port per sampler shard (several islands at once)
            port = _shard_port(port, spec.get("sampler_id"))
        tunnel = TunnelSpec(listen_host=host, listen_port=_port(port, "agentic.setup.tunnel.listen port"),
                            target_port=_port(target, "agentic.setup.tunnel.target_port"),
                            user=str(tn.get("user", sandbox.user)), host=str(tn.get("host", sandbox.host)),
                            port=_port(tn.get("port", sandbox.port), "agentic.setup.tunnel.port"), key=key,
                            settle_s=float(tn.get("settle_s", 5)), retry_s=float(tn.get("retry_s", 2)),
                            max_backoff_s=float(tn.get("max_backoff_s", 60)),
                            ready_timeout_s=float(tn.get("ready_timeout_s", 90)))
        if public := os.environ.get("YETO_RECORDS_PROXY_PUBLIC_BASE"):
            notes.append(f"YETO_RECORDS_PROXY_PUBLIC_BASE={public} in this environment is not used: the setup "
                         "sets it from tunnel.listen")
    else:
        notes.append("no tunnel section: the sandboxes reach the records proxy some other way")

    tasks = None
    ts = _section(doc, "tasks")
    if ts:
        source = str(ts.get("source") or "")
        if not source:
            raise ValueError("agentic.setup.tasks.source is required (a local directory or store:<prefix>)")
        if ts.get("ids") and ts.get("ids_file"):
            raise ValueError("agentic.setup.tasks: give ids or ids_file, not both")
        if ts.get("ids_file"):
            ids_path = Path(local(str(ts["ids_file"])))
            if not ids_path.is_file():
                raise ValueError(f"agentic.setup.tasks.ids_file {ids_path} does not exist")
            ids = [ln.strip() for ln in ids_path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        else:
            ids = [str(i) for i in ts.get("ids") or ()]
        bad = [i for i in ids if not _NAME.match(i)]
        if bad:
            raise ValueError(f"agentic.setup.tasks: bad task ids {bad[:5]}")
        if not source.startswith(STORE_PREFIX):
            source = local(source)
        elif not source[len(STORE_PREFIX):].strip("/"):
            raise ValueError("agentic.setup.tasks.source store: needs a prefix")
        tasks = TasksSpec(source=source, ids=tuple(ids))

    secret: SecretRef | str | None = None
    if "records_secret" in doc:
        raw = doc["records_secret"]
        secret = "generate" if raw == "generate" else ref(raw, "agentic.setup.records_secret")
    elif tunnel is not None:
        raise ValueError("agentic.setup: a tunnel makes the records proxy public; set records_secret "
                         "(\"generate\", {\"path\"} or {\"env\"})")

    provider = ((agentic.get("task_defaults") or {}).get("provider"))
    if provider and provider != "remote-docker":
        raise ValueError(f"agentic.setup prepares remote-docker sandboxes; task_defaults.provider is {provider!r}")
    return AgenticSetup(root=root, harbor_version=version, harbor_commit=commit, harbor_keep=keep,
                        docker_cli_version=cli_version, compose_version=compose, sandbox=sandbox, tunnel=tunnel,
                        tasks=tasks, records_secret=secret, notes=tuple(notes),
                        sandbox_host=None if envs_host is None else envs_host.name)


def _envs(spec: Mapping[str, Any], agentic: Mapping[str, Any]) -> Any:
    from yeto.sandbox.envs import EnvsError, parse_envs  # noqa: PLC0415

    section = agentic.get("envs")
    if not section:
        return None
    try:
        return parse_envs(section, run=str(section.get("run") or spec.get("run_prefix") or "run"))
    except EnvsError as error:
        raise ValueError(str(error)) from None


def _envs_host(spec: Mapping[str, Any], agentic: Mapping[str, Any]) -> Any:
    """The sandbox host this sampler drives (``docker_host: "envs"``)."""
    from yeto.sandbox.envs import EnvsError, assign_host  # noqa: PLC0415

    cfg = _envs(spec, agentic)
    if cfg is None or not cfg.hosts:
        raise ValueError("agentic.setup.sandbox.docker_host \"envs\" needs agentic.envs.hosts")
    try:
        return assign_host(cfg, spec.get("sampler_id"))
    except EnvsError as error:
        raise ValueError(f"agentic.setup.sandbox.docker_host \"envs\": {error}") from None


def _envs_host_of(spec: Mapping[str, Any], agentic: Mapping[str, Any], url: Any) -> Any:
    """The ``agentic.envs.hosts`` entry of an explicit ``docker_host`` (None: no envs section,
    or no entry for it), so the gate can still find the sampler's host."""
    try:
        cfg = _envs(spec, agentic)
    except ValueError:
        return None
    if cfg is None or not url.hostname:
        return None
    for h in cfg.hosts:
        u = urlsplit(h.docker_host)
        if u.hostname == url.hostname and (u.port or 22) == (url.port or 22):
            return h
    return None


def local_problems(setup: AgenticSetup, *, environ: Mapping[str, str] | None = None) -> list[str]:
    """What the launcher host is missing to run the setup (files, modes, variables)."""
    environ = os.environ if environ is None else environ
    out: list[str] = []
    for what, r, private in (("sandbox.key", setup.sandbox.key, True),
                             ("sandbox.known_hosts", setup.sandbox.known_hosts, False),
                             ("tunnel.key", setup.tunnel.key if setup.tunnel else None, True)):
        if r is not None and (p := r.problem(environ, private=private)):
            out.append(f"agentic.setup.{what}: {p}")
    if isinstance(setup.records_secret, SecretRef):
        if p := setup.records_secret.problem(environ, private=True):
            out.append(f"agentic.setup.records_secret: {p}")
        else:
            try:
                if len(setup.records_secret.read(environ).strip()) < MIN_SECRET_CHARS:
                    out.append(f"agentic.setup.records_secret: shorter than {MIN_SECRET_CHARS} characters")
            except OSError as error:
                out.append(f"agentic.setup.records_secret: {error}")
    if setup.tasks and setup.tasks.store_prefix is None:
        src = Path(setup.tasks.source)
        if not src.is_dir() and staged_by_sandbox_prepare(src):
            pass  # `yeto sandbox prepare` (run up's step after the preflight) stages it
        elif not src.is_dir():
            out.append(f"agentic.setup.tasks.source {src} is not a directory")
        else:
            missing = [i for i in setup.tasks.ids if not (src / i / "task.toml").is_file()]
            if missing:
                out.append(f"agentic.setup.tasks: no task.toml under {src} for {missing[:10]}"
                           f"{' ...' if len(missing) > 10 else ''}")
            if not setup.tasks.ids and not any(src.glob("*/task.toml")):
                out.append(f"agentic.setup.tasks.source {src} holds no <id>/task.toml")
    return out


def staged_by_sandbox_prepare(path: Path) -> bool:
    """``path`` lies under ``yeto sandbox prepare``'s default stage dir
    (``~/.yeto/sandbox/<run>/stage``): a first dry run of a new run comes before it exists."""
    root = Path("~/.yeto/sandbox").expanduser().resolve()
    try:
        rel = Path(path).expanduser().resolve().relative_to(root)
    except ValueError:
        return False
    return len(rel.parts) >= 2 and rel.parts[1] == "stage"


def describe(setup: AgenticSetup) -> list[str]:
    """The plan, one line per step (``--dry-run``)."""
    s, t = setup.sandbox, setup.tunnel
    lines = [f"agentic setup under {setup.root} (node: <nvme>{setup.root})",
             f"  docker_cli: docker {setup.docker_cli_version} + compose v{setup.compose_version} in {setup.root}/bin",
             f"  harbor: {setup.harbor_version} @ {setup.harbor_commit[:12]} into {setup.site} "
             f"(keep {list(setup.harbor_keep)})",
             f"  sandbox: DOCKER_HOST=ssh://{s.alias} -> {s.user}@{s.host}:{s.port}"
             + (f" (agentic.envs host {setup.sandbox_host})" if setup.sandbox_host else "")
             + f", key {s.key.describe()}, "
             f"known_hosts {s.known_hosts.describe() if s.known_hosts else 'accept-new'}"]
    if t:
        lines.append(f"  tunnel: -R {t.listen} -> 127.0.0.1:{t.target_port} via {t.user}@{t.host}:{t.port}, "
                     f"key {t.key.describe()}, container {TUNNEL_CONTAINER} (restart unless-stopped, "
                     f"reconnect backoff {t.retry_s:g}..{t.max_backoff_s:g} s), events {setup.tunnel_events}")
    if setup.tasks:
        lines.append(f"  tasks: {len(setup.tasks.ids) or 'all'} task(s) from {setup.tasks.source} "
                     f"-> {setup.root}/tasks")
    if setup.records_secret is not None:
        src = setup.records_secret if isinstance(setup.records_secret, str) else setup.records_secret.describe()
        lines.append(f"  records_secret: {src} -> {setup.secret_file} (0600)")
    lines.append("  sampler env: " + " ".join(f"{k}={v}" for k, v in sorted(setup.container_env().items())
                                               if k != "PATH"))
    lines += [f"  note: {n}" for n in setup.notes]
    return lines


def dry_run_report(spec: Mapping[str, Any], *, base_dir: str | Path | None = None,
                   environ: Mapping[str, str] | None = None) -> tuple[list[str], list[str]]:
    """``(lines, problems)`` of a spec's setup section for a dry run (``pool --dry-run``,
    ``agentic-setup --dry-run``); no section gives ``([], [])``."""
    try:
        setup = parse(spec, base_dir=base_dir)
    except ValueError as error:
        return [], [str(error)]
    if setup is None:
        return [], []
    return describe(setup), local_problems(setup, environ=environ)


def strip(spec: Mapping[str, Any]) -> dict[str, Any]:
    """``spec`` without ``agentic.setup`` (local paths stay on the launcher)."""
    out = dict(spec)
    if isinstance(out.get("agentic"), Mapping) and "setup" in out["agentic"]:
        out["agentic"] = {k: v for k, v in out["agentic"].items() if k != "setup"}
    return out


# -- the node --------------------------------------------------------------------

@dataclass
class SshNode:
    """A sampler node over SSH (``alias``), with the sampler image and the yeto tree
    uploaded for this start. ``prelude`` runs first in every helper container
    (``ssh_sampler.SSH_HOME_LINK``: OpenSSH reads the passwd home's ``.ssh``)."""

    alias: str
    image: str
    nvme: str
    yeto_dir: str
    python: str
    user: str
    prelude: str = ""
    run: Callable[..., Any] = subprocess.run

    def sh(self, command: str, *, timeout: float = 600, input: str | None = None):
        argv = ["ssh", "-T", "-o", "BatchMode=yes", self.alias, command]
        if input is None:
            argv.insert(1, "-n")
        return self.run(argv, capture_output=True, text=True, timeout=timeout, check=False, input=input)

    def host(self, container_path: str) -> str:
        return self.nvme + container_path

    def put(self, text: str, remote: str, *, mode: str = "600") -> bool:
        """Write ``text`` to ``remote`` (host path) as root with ``mode``; False when it was there already."""
        digest = hashlib.sha256(text.encode()).hexdigest()
        q = shlex.quote(remote)
        out = self.sh(f"sudo sha256sum {q} 2>/dev/null | cut -d' ' -f1; sudo stat -c %a {q} 2>/dev/null",
                      timeout=60)
        have = (out.stdout or "").split()
        if have[:2] == [digest, mode]:
            return False
        out = self.sh(f"sudo mkdir -p {shlex.quote(os.path.dirname(remote))} && sudo sh -c "
                      + shlex.quote(f"umask 077; cat > {q}.tmp && chmod {mode} {q}.tmp && mv -f {q}.tmp {q}"),
                      timeout=120, input=text)
        if out.returncode != 0:
            raise SetupError("upload", f"writing {remote} on the node: {(out.stderr or '').strip()[-300:]}")
        return True

    def in_image(self, script: str, *, env: Mapping[str, str] | None = None, timeout: float = 900):
        argv = ["sudo", "docker", "run", "--rm", "--network", "host", "-v", f"{self.nvme}/work:/work",
                "--entrypoint", "bash"]
        for k, v in sorted((env or {}).items()):
            argv += ["-e", f"{k}={v}"]
        return self.sh(shlex.join([*argv, self.image, "-c", self.prelude + script]), timeout=timeout)

    def stream_tar(self, src: str, names: Sequence[str], remote_dir: str, *, timeout: float = 1800) -> None:
        tar = subprocess.Popen(["tar", "-C", src, "-cf", "-", *(names or ["."])], stdout=subprocess.PIPE)
        try:
            out = self.run(["ssh", "-T", "-o", "BatchMode=yes", self.alias,
                            f"sudo mkdir -p {shlex.quote(remote_dir)} && sudo tar -C {shlex.quote(remote_dir)} "
                            "-xf - --no-same-owner"], stdin=tar.stdout, capture_output=True, text=True,
                           timeout=timeout, check=False)
        finally:
            tar.wait()
        if tar.returncode != 0 or out.returncode != 0:
            raise SetupError("tasks", f"copying {src} to the node failed: {(out.stderr or '').strip()[-300:]}")


def key_identity(ref: SecretRef, *, environ: Mapping[str, str] | None = None,
                 run: Callable[..., Any] = subprocess.run) -> tuple[str, str]:
    """``(fingerprint, public key line)`` of a private key (``ssh-keygen`` on the launcher)."""
    environ = os.environ if environ is None else environ
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "key"
        path.write_text(ref.read(environ))
        path.chmod(0o600)
        fpr = run(["ssh-keygen", "-lf", str(path)], capture_output=True, text=True, check=False)
        pub = run(["ssh-keygen", "-y", "-f", str(path)], capture_output=True, text=True, check=False)
    fingerprint = (fpr.stdout or "").split()[1] if len((fpr.stdout or "").split()) > 1 else "unknown"
    return fingerprint, (pub.stdout or "").strip()


def classify_ssh_failure(text: str, *, who: str, key: str, fingerprint: str, purpose: str) -> str:
    """An operator message for an ssh / docker-over-ssh failure output."""
    low = text.lower()
    if "permission denied" in low and "publickey" in low:
        return (f"the sandbox host refused the {key} ({who}): add the public key with fingerprint {fingerprint} "
                f"to {who.split('@')[0]}'s authorized_keys on the sandbox host ({purpose}; "
                "see docs/KIMI_K31_ASYNC_RL_GUIDE.md, Sandbox host)")
    if "remote port forwarding failed" in low or "forwarding failed" in low:
        return (f"the sandbox host refused the reverse forward ({who}): the tunnel key's authorized_keys line "
                "needs permitlisten for this listen address (one line per key: sshd uses the first that "
                "matches) and sshd needs GatewayPorts clientspecified (docs/KIMI_K31_ASYNC_RL_GUIDE.md, Sandbox host); or the port is taken by "
                "another tunnel")
    if "host key verification failed" in low or "remote host identification has changed" in low:
        return (f"the host key of {who} does not match the pinned known_hosts: refresh "
                "agentic.setup.sandbox.known_hosts from the sandbox host")
    if "connection refused" in low:
        return f"{who}: connection refused (sshd down or a firewall rejects this node's egress IP)"
    if "timed out" in low or "no route to host" in low or "network is unreachable" in low:
        return f"{who}: unreachable from the node (firewall allowlist, routing, or the host is down)"
    if "could not resolve" in low or "name or service not known" in low:
        return f"{who}: the host name does not resolve on the node"
    if "cannot connect to the docker daemon" in low or "docker: command not found" in low:
        return (f"{who} accepted the key but has no docker daemon for this user (rootless docker of the "
                "sandbox user, docs/KIMI_K31_ASYNC_RL_GUIDE.md, Sandbox host)")
    return f"{who}: {text.strip()[-300:] or 'failed without output'}"


# -- provisioning ----------------------------------------------------------------

HARBOR_CHECK = r"""
import json, sys
from importlib import metadata
out = {}
try:
    d = metadata.distribution("harbor")
    out["version"] = d.version
    raw = d.read_text("direct_url.json")
    out["commit"] = (json.loads(raw).get("vcs_info") or {}).get("commit_id") if raw else None
    import harbor, vllm  # noqa: F401  Harbor is imported in-process next to vLLM
    out["import"] = "ok"
except Exception as error:  # noqa: BLE001
    out["error"] = f"{type(error).__name__}: {error}"[:400]
print("YETO_SETUP_HARBOR " + json.dumps(out))
"""

HARBOR_PRUNE = r"""
import shutil, sys
from importlib import metadata
from pathlib import Path
site, keep = Path(sys.argv[1]), set(sys.argv[2].split(",")) | {"harbor"}
norm = lambda n: n.lower().replace("_", "-")
base = {norm(d.metadata["Name"]) for d in metadata.distributions(path=[p for p in sys.path if p and str(site) not in p])}
pruned = []
for d in metadata.distributions(path=[str(site)]):
    name = norm(d.metadata["Name"])
    if name in base and name not in keep:
        for f in d.files or ():
            p = site / f
            if p.is_file() or p.is_symlink():
                p.unlink()
        shutil.rmtree(Path(d._path), ignore_errors=True)
        pruned.append(name)
for p in sorted(site.iterdir()):  # package dirs left empty by the removals
    if p.is_dir() and not any(f.is_file() for f in p.rglob("*")):
        shutil.rmtree(p, ignore_errors=True)
print("YETO_SETUP_PRUNED " + ",".join(sorted(pruned)))
"""


def _ssh_config(setup: AgenticSetup) -> str:
    s, home = setup.sandbox, setup.home
    return "\n".join([
        f"Host {s.alias}", f"  HostName {s.host}", f"  User {s.user}", f"  Port {s.port}",
        f"  IdentityFile {home}/.ssh/{s.alias}.key", "  IdentitiesOnly yes", "  BatchMode yes",
        f"  StrictHostKeyChecking {'yes' if s.known_hosts else 'accept-new'}",
        f"  UserKnownHostsFile {home}/.ssh/known_hosts", "  ControlMaster auto",
        "  ControlPath /tmp/yeto-ssh-%C", "  ControlPersist 10m", "  ServerAliveInterval 15", ""])


def _tunnel_config(setup: AgenticSetup) -> str:
    t = setup.tunnel
    assert t is not None
    return "\n".join([
        f"Host {TUNNEL_ALIAS}", f"  HostName {t.host}", f"  User {t.user}", f"  Port {t.port}",
        "  IdentityFile /tunnel/key", "  IdentitiesOnly yes", "  BatchMode yes",
        f"  StrictHostKeyChecking {'yes' if setup.sandbox.known_hosts else 'accept-new'}",
        "  UserKnownHostsFile /tunnel/known_hosts", "  ExitOnForwardFailure yes",
        "  ServerAliveInterval 15", "  ServerAliveCountMax 3", ""])


def tunnel_ssh_argv(setup: AgenticSetup) -> list[str]:
    t = setup.tunnel
    assert t is not None
    return ["ssh", "-N", "-F", "/tunnel/config", "-R", f"{t.listen}:127.0.0.1:{t.target_port}", TUNNEL_ALIAS]


def tunnel_run_argv(setup: AgenticSetup, node: SshNode) -> list[str]:
    """``docker run`` of the tunnel helper: host network, the tunnel key directory
    (outside ``/work``) and the events directory only."""
    t = setup.tunnel
    assert t is not None
    events_dir = node.host(f"{setup.root}/events")
    return ["sudo", "docker", "run", "-d", "--name", TUNNEL_CONTAINER, "--restart", "unless-stopped",
            "--network", "host", "--label", f"yeto.setup={setup.fingerprint()}",
            "-v", f"{node.yeto_dir}:/yeto:ro", "-v", f"{node.nvme}/agentic-tunnel:/tunnel",
            "-v", f"{events_dir}:/events", "-e", "PYTHONPATH=/yeto", "--entrypoint", node.python, node.image,
            "-m", "yeto.launch.agentic_node_agent", "tunnel", "--events", "/events/tunnel.jsonl",
            "--settle-s", f"{t.settle_s:g}", "--retry-s", f"{t.retry_s:g}", "--max-backoff-s",
            f"{t.max_backoff_s:g}", "--", *tunnel_ssh_argv(setup)]


def _events(text: str) -> list[dict[str, Any]]:
    out = []
    for line in (text or "").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


class Provisioner:
    """Runs the steps of one :class:`AgenticSetup` on one :class:`SshNode`."""

    def __init__(self, setup: AgenticSetup, node: SshNode, *, emit: Callable[[dict[str, Any]], None] | None = None,
                 environ: Mapping[str, str] | None = None, store_env_file: str | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 local_run: Callable[..., Any] = subprocess.run):
        self.setup, self.node = setup, node
        self.emit = emit or (lambda record: None)
        self.environ = os.environ if environ is None else environ
        self.store_env_file = store_env_file
        self.sleep, self.clock, self.local_run = sleep, clock, local_run
        self.report: dict[str, str] = {}

    def _done(self, step: str, status: str, detail: str = "") -> None:
        self.report[step] = status
        self.emit({"event": SETUP_EVENT, "step": step, "status": status, "node": self.node.alias,
                   **({"detail": detail} if detail else {})})

    def _fail(self, step: str, message: str) -> SetupError:
        self.report[step] = "failed"
        self.emit({"event": SETUP_EVENT, "step": step, "status": "failed", "node": self.node.alias,
                   "detail": message})
        return SetupError(step, message)

    def _image_env(self) -> dict[str, str]:
        env = self.setup.container_env(python_bin=os.path.dirname(self.node.python))
        return {"HOME": env["HOME"], "PATH": env["PATH"], "PYTHONPATH": self.setup.site}

    def run(self) -> dict[str, str]:
        """Every step in order; returns ``{step: ok|changed}``. Raises :class:`SetupError`."""
        local = local_problems(self.setup, environ=self.environ)
        if local:
            raise self._fail("local", "; ".join(local))
        self.layout()
        self.docker_cli()
        self.harbor()
        self.sandbox()
        self.records_secret()
        self.tasks()
        self.tunnel()
        return dict(self.report)

    # -- steps
    def layout(self) -> None:
        r = shlex.quote(self.node.host(self.setup.root))
        out = self.node.sh(f"sudo mkdir -p {r}/bin {r}/site {r}/home/.ssh {r}/home/.docker/cli-plugins {r}/tasks "
                           f"{r}/trials {r}/events && sudo chmod 700 {r}/home/.ssh && sudo mkdir -p "
                           f"{shlex.quote(self.tunnel_dir)} && sudo chmod 700 "
                           f"{shlex.quote(self.tunnel_dir)}", timeout=120)
        if out.returncode != 0:
            raise self._fail("layout", f"creating {self.node.host(self.setup.root)}: {(out.stderr or '')[-300:]}")
        self._done("layout", "ok")

    def _docker_versions(self) -> tuple[str, str]:
        out = self.node.in_image("docker --version 2>&1; docker compose version 2>&1", env=self._image_env(),
                                 timeout=300)
        text = out.stdout or ""
        self._last_image_error = "" if out.returncode == 0 else (out.stderr or out.stdout or "").strip()[-300:]
        cli = re.search(r"Docker version (\d+\.\d+\.\d+)", text)
        compose = re.search(r"Docker Compose version v?(\d+\.\d+\.\d+)", text)
        return (cli.group(1) if cli else ""), (compose.group(1) if compose else "")

    def docker_cli(self) -> None:
        want = (self.setup.docker_cli_version, self.setup.compose_version)
        if self._docker_versions() == want:
            return self._done("docker_cli", "ok")
        r = self.node.host(self.setup.root)
        b, plug = shlex.quote(f"{r}/bin"), shlex.quote(f"{r}/home/.docker/cli-plugins/docker-compose")
        out = self.node.sh(
            "set -eu; "
            f"curl -fsSL --retry 3 {shlex.quote(DOCKER_STATIC_URL.format(v=want[0]))} | sudo tar xz "
            f"--strip-components=1 -C {b} docker/docker; "
            f"sudo curl -fsSL --retry 3 -o {plug} {shlex.quote(COMPOSE_URL.format(v=want[1]))}; "
            f"sudo chmod 755 {b}/docker {plug}", timeout=900)
        if out.returncode != 0:
            raise self._fail("docker_cli", f"download of docker {want[0]} / compose v{want[1]} failed on the node: "
                                           f"{(out.stderr or '').strip()[-300:]}")
        have = self._docker_versions()
        if have != want:
            raise self._fail("docker_cli", f"after the install the sampler image runs docker {have[0] or '-'} / "
                                           f"compose {have[1] or '-'}, wanted {want[0]} / {want[1]}"
                                           + (f"; the sampler image run failed: {self._last_image_error}"
                                              if getattr(self, "_last_image_error", "") else ""))
        self._done("docker_cli", "changed", f"docker {want[0]}, compose v{want[1]}")

    def _harbor_state(self) -> dict[str, Any]:
        out = self.node.in_image(f"{shlex.quote(self.node.python)} -c {shlex.quote(HARBOR_CHECK)}",
                                 env=self._image_env(), timeout=600)
        for line in (out.stdout or "").splitlines():
            if line.startswith("YETO_SETUP_HARBOR "):
                return json.loads(line.split(" ", 1)[1])
        return {"error": (out.stderr or out.stdout or "no output").strip()[-300:]}

    def _harbor_ok(self, state: Mapping[str, Any]) -> bool:
        return (state.get("version") == self.setup.harbor_version and state.get("commit") == self.setup.harbor_commit
                and state.get("import") == "ok")

    def harbor(self) -> None:
        if self._harbor_ok(self._harbor_state()):
            return self._done("harbor", "ok")
        req = f"harbor @ git+{HARBOR_REPO}@{self.setup.harbor_commit}"
        # Harbor depends on its uv workspace member server-dashboard (not on PyPI): install it from
        # the same commit's subdirectory in the same resolve
        dash = f"server-dashboard @ git+{HARBOR_REPO}@{self.setup.harbor_commit}#subdirectory=server-dashboard"
        py, site = shlex.quote(self.node.python), shlex.quote(self.setup.site)
        out = self.node.in_image(
            f"set -e; {py} -m pip install --quiet --disable-pip-version-check --upgrade --target {site} "
            f"{shlex.quote(req)} {shlex.quote(dash)}; PYTHONPATH= {py} -c {shlex.quote(HARBOR_PRUNE)} {site} "
            f"{shlex.quote(','.join(self.setup.harbor_keep))}", env=self._image_env(), timeout=1800)
        if out.returncode != 0:
            raise self._fail("harbor", f"pip install of Harbor {self.setup.harbor_version} into {self.setup.site} "
                                       f"failed: {(out.stderr or out.stdout or '').strip()[-400:]}")
        state = self._harbor_state()
        if not self._harbor_ok(state):
            raise self._fail("harbor", f"Harbor in {self.setup.site} is not usable next to vLLM: {state} (wanted "
                                       f"{self.setup.harbor_version} @ {self.setup.harbor_commit[:12]}; a package "
                                       "conflict may need harbor.keep)")
        pruned = next((ln.split(" ", 1)[1] for ln in (out.stdout or "").splitlines()
                       if ln.startswith("YETO_SETUP_PRUNED ")), "")
        self._done("harbor", "changed", f"pruned {len([p for p in pruned.split(',') if p])} package(s)")

    def sandbox(self) -> None:
        s, ssh_dir = self.setup.sandbox, self.node.host(f"{self.setup.home}/.ssh")
        changed = self.node.put(s.key.read(self.environ), f"{ssh_dir}/{s.alias}.key")
        changed |= self.node.put(_ssh_config(self.setup), f"{ssh_dir}/config")
        if s.known_hosts:
            changed |= self.node.put(s.known_hosts.read(self.environ), f"{ssh_dir}/known_hosts", mode="644")
        out = self.node.in_image(f"docker -H ssh://{s.alias} version --format '{{{{.Server.Version}}}}' 2>&1 && "
                                 f"docker -H ssh://{s.alias} info --format '{{{{.SecurityOptions}}}}' 2>&1",
                                 env=self._image_env(), timeout=180)
        text = (out.stdout or "") + (out.stderr or "")
        if out.returncode != 0:
            fpr, _ = key_identity(s.key, environ=self.environ, run=self.local_run)
            raise self._fail("sandbox", classify_ssh_failure(text, who=f"{s.user}@{s.host}:{s.port}",
                                                             key="docker key", fingerprint=fpr,
                                                             purpose="the docker dial-only key"))
        server = text.strip().splitlines()[0] if text.strip() else "?"
        detail = f"daemon {server}" + ("" if "rootless" in text else " (not rootless: see docs/KIMI_K31_ASYNC_RL_GUIDE.md, Sandbox host)")
        self._done("sandbox", "changed" if changed else "ok", detail)

    def records_secret(self) -> None:
        ref = self.setup.records_secret
        if ref is None:
            return
        path = self.node.host(self.setup.secret_file)
        q = shlex.quote(path)
        changed = False
        if isinstance(ref, SecretRef):
            changed = self.node.put(ref.read(self.environ), path)
        else:  # generate once; a rerun keeps the node's secret
            out = self.node.sh("sudo sh -c " + shlex.quote(
                f"test -s {q} && echo kept || (umask 077; head -c 48 /dev/urandom | base64 | tr -d '\\n/+=' > {q}; "
                "echo made)"), timeout=60)
            changed = "made" in (out.stdout or "")
        # the redirect must run as root too (the file is 0600 root): one root shell for both
        out = self.node.sh("sudo sh -c " + shlex.quote(f"stat -c %a {q}; wc -c < {q}"), timeout=60)
        parts = (out.stdout or "").split()
        if out.returncode != 0 or len(parts) < 2 or parts[0] != "600" or int(parts[1]) < MIN_SECRET_CHARS:
            raise self._fail("records_secret", f"{path} must be mode 0600 with at least {MIN_SECRET_CHARS} "
                                               f"characters, found {parts or 'nothing'}")
        self._done("records_secret", "changed" if changed else "ok")

    def _tasks_digest(self, src: Path, ids: Sequence[str]) -> str:
        h = hashlib.sha256()
        roots = [src / i for i in ids] if ids else [src]
        for root in roots:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    st = p.stat()
                    h.update(f"{p.relative_to(src)}\0{st.st_size}\0{int(st.st_mtime)}\n".encode())
        return h.hexdigest()[:16]

    def tasks(self) -> None:
        t = self.setup.tasks
        if t is None:
            return
        dest = self.node.host(f"{self.setup.root}/tasks")
        changed = False
        if t.store_prefix is None:
            src = Path(t.source)
            stamp = f"{dest}/.yeto-setup-{self._tasks_digest(src, t.ids)}"
            if self.node.sh(f"sudo test -f {shlex.quote(stamp)}", timeout=60).returncode != 0:
                self.node.stream_tar(str(src), list(t.ids), dest)
                self.node.sh(f"sudo rm -f {shlex.quote(dest)}/.yeto-setup-*; sudo touch {shlex.quote(stamp)}",
                             timeout=60)
                changed = True
        else:
            if not self.store_env_file:
                raise self._fail("tasks", "a store: task source needs the store env file on the node "
                                          "(--r2-env-file)")
            keys = [f"{t.store_prefix}/{i}" for i in t.ids] or [t.store_prefix]
            into = self._store_sync_root(dest)
            body = " && ".join(
                f'aws s3 sync --only-show-errors "s3://$R2_BUCKET/{k}/" {into}/{shlex.quote(k[len(t.store_prefix):].strip("/"))} '
                '--endpoint-url "$AWS_ENDPOINT_URL"' for k in keys)
            out = self.node.sh(self._store_sync_command(dest, body), timeout=1800)
            if out.returncode != 0:
                raise self._fail("tasks", f"store sync of {t.source} failed: {(out.stderr or '').strip()[-300:]}")
            changed = True
        if t.ids:
            check = "; ".join(f"sudo test -f {shlex.quote(dest)}/{i}/task.toml || echo MISSING {i}" for i in t.ids)
            out = self.node.sh(check, timeout=120)
            missing = [ln.split()[1] for ln in (out.stdout or "").splitlines() if ln.startswith("MISSING ")]
            if missing:
                raise self._fail("tasks", f"no task.toml on the node for {missing[:10]} under {dest} "
                                          f"(source {t.source})")
        self._done("tasks", "changed" if changed else "ok", f"{len(t.ids) or 'all'} task(s)")

    # -- the docker-specific pieces (a subclass without docker replaces them)
    @property
    def tunnel_dir(self) -> str:
        """The tunnel key directory on the node (outside the sampler's ``/work``)."""
        return f"{self.node.nvme}/agentic-tunnel"

    def _store_sync_root(self, dest: str) -> str:  # noqa: ARG002 (dest is mounted at /tasks)
        return "/tasks"

    def _store_sync_command(self, dest: str, body: str) -> str:
        return (f"sudo docker run --rm --network host --env-file {shlex.quote(str(self.store_env_file))} "
                f"-v {shlex.quote(dest)}:/tasks --entrypoint sh {AWS_IMAGE} -c {shlex.quote(body)}")

    def _tunnel_running(self) -> bool:
        out = self.node.sh(f"sudo docker inspect -f '{{{{.State.Running}}}} {{{{index .Config.Labels "
                           f"\"yeto.setup\"}}}}' {TUNNEL_CONTAINER} 2>/dev/null || true", timeout=60)
        return (out.stdout or "").strip() == f"true {self.setup.fingerprint()}"

    def _tunnel_start(self) -> Any:
        return self.node.sh(f"sudo docker rm -f {TUNNEL_CONTAINER} >/dev/null 2>&1; "
                            + shlex.join(tunnel_run_argv(self.setup, self.node)), timeout=300)

    def _tunnel_lines(self) -> list[dict[str, Any]]:
        out = self.node.sh(f"sudo tail -n 50 {shlex.quote(self.node.host(self.setup.tunnel_events))} 2>/dev/null",
                           timeout=60)
        return [e for e in _events(out.stdout) if e.get("event") == TUNNEL_EVENT]

    def tunnel(self) -> None:
        t = self.setup.tunnel
        if t is None:
            return
        tdir = self.tunnel_dir
        changed = self.node.put(t.key.read(self.environ), f"{tdir}/key")
        changed |= self.node.put(_tunnel_config(self.setup), f"{tdir}/config")
        if self.setup.sandbox.known_hosts:
            changed |= self.node.put(self.setup.sandbox.known_hosts.read(self.environ), f"{tdir}/known_hosts",
                                     mode="644")
        running = self._tunnel_running()
        before = len(self._tunnel_lines()) if not running or changed else None
        if not running or changed:
            out = self._tunnel_start()
            if out.returncode != 0:
                raise self._fail("tunnel", f"starting {TUNNEL_CONTAINER}: {(out.stderr or '').strip()[-300:]}")
        deadline = self.clock() + t.ready_timeout_s
        while True:
            lines = self._tunnel_lines()
            fresh = lines if before is None else lines[before:] if len(lines) >= before else lines
            last = fresh[-1] if fresh else None
            if last and last.get("state") == "up":
                return self._done("tunnel", "ok" if before is None else "changed",
                                  f"-R {t.listen} -> 127.0.0.1:{t.target_port}")
            downs = [e for e in fresh if e.get("state") == "down"]
            if downs and self.clock() >= deadline or len(downs) >= 3:
                fpr, _ = key_identity(t.key, environ=self.environ, run=self.local_run)
                raise self._fail("tunnel", classify_ssh_failure(str(downs[-1].get("stderr") or ""),
                                                                who=f"{t.user}@{t.host}:{t.port}", key="tunnel key",
                                                                fingerprint=fpr,
                                                                purpose=f"the tunnel key, permitlisten {t.listen}"))
            if self.clock() >= deadline:
                raise self._fail("tunnel", f"no tunnel up event in {t.ready_timeout_s:g} s "
                                           f"({self.setup.tunnel_events} on the node)")
            self.sleep(2.0)


def tunnel_events(node: SshNode, setup: AgenticSetup, *, since: int = 0) -> tuple[list[dict[str, Any]], int]:
    """The tunnel's up/down events on the node after line ``since``; returns them and
    the new line count (a watcher passes it back as ``since``)."""
    out = node.sh(f"sudo cat {shlex.quote(node.host(setup.tunnel_events))} 2>/dev/null", timeout=120)
    lines = (out.stdout or "").splitlines()
    return [e for e in _events("\n".join(lines[since:])) if e.get("event") == TUNNEL_EVENT], len(lines)
