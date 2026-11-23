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

AgentENV microVMs (``"provider": "agentenv"``, or ``task_defaults.provider``
``agentenv``): there is no docker host, so ``sandbox``, ``tunnel`` and
``docker_cli`` are refused and their steps do not run. The section is::

    "setup": {
      "provider": "agentenv",
      "agentenv": {"gateway_url": "https://<gateway>",                       # required
                   "sandbox_url": "https://<public data-plane gateway>",      # default: gateway_url
                   "api_key": {"path": "<local env-file>"} | {"env": "<VAR>"},  # required
                   "machine_type": "<type>", "max_sandboxes": 16, "pool": true},
      "records_proxy": {"public_base": "https://<public records proxy base>"},
      "tasks": {...}, "records_secret": "generate"
    }

Steps: ``layout``, ``harbor`` (plus the pinned ``e2b`` SDK), ``records_secret``,
``tasks``, ``agentenv_key`` (the key file ``agentenv.key`` under ``<root>/home``, 0600;
the container reads it through ``AGENTENV_API_KEY_FILE``, so the key is in no env
file and no dry-run line). The microVMs deny private egress: the records proxy
must be reached over a public HTTPS base; a missing or placeholder
``records_proxy.public_base`` is a warning in a dry run and a failure at setup.

A sampler node that is reachable on port 22 only (a National Compute node: a
private address behind a 1:1 NAT whose cloud firewall passes 22 alone) cannot
serve the records proxy at ``http://<node ip>:<port>``: the microVMs' SYNs never
arrive and every trial fails with no recorded model call (a silent zero reward).
``records_relay`` (instead of ``records_proxy``) puts the proxy behind a relay
host the microVMs reach::

      "records_relay": {"host": "<relay ssh host>",                            # required
                        "user": "root", "port": 22,
                        "key": {"path": "<local key; {shard} -> shard index>"} | {"env": ...},  # required
                        "known_hosts": {"path": ...} | {"env": ...},
                        "listen_port": <port> | "<lo>-<hi>",                   # required; shard k -> lo + k
                        "public_host": "<address the microVMs dial>",          # default: host
                        "bind": "0.0.0.0",
                        "allow_cidrs": ["<sandbox egress cidr>", ...],         # relay firewall (records-relay)
                        "settle_s": 5, "ready_timeout_s": 60}

Step ``records_relay``: the key (and known_hosts) under ``<nvme>/records-relay``
(0700, outside the sampler's ``/work``), and a systemd unit
``yeto-records-relay.service`` (``Restart=always``) running ``ssh -N
-o ExitOnForwardFailure=yes -R <bind>:<listen_port>:127.0.0.1:<agentic.proxy.port>``
to the relay; checked by the unit staying active without a restart for
``settle_s``. A host without systemd (PID 1 is not systemd, or no ``systemctl``:
a Daytona or Modal sandbox, most containers) runs the same forward under yeto's
reconnecting supervisor instead (``python -m yeto.launch.agentic_node_agent
tunnel``: reconnect with backoff, one JSON line per up / down in
``<relay dir>/events.jsonl``): a ``docker run --restart unless-stopped`` helper
container on a docker node, a detached process on a process node (its PID in
``<relay dir>/supervisor.pid``; stopped by that PID after a check of its command
line, never by a name match). Either way every sampler start runs this step
again, so a rebooted host gets its forward back. The sampler's ``YETO_RECORDS_PROXY_PUBLIC_BASE`` becomes
``http://<public_host>:<listen_port>``. The relay host side (sshd ``GatewayPorts
clientspecified``, the restricted ``authorized_keys`` line of every node key,
the firewall of the listen ports) is one idempotent command:
``python -m yeto.launch.nc records-relay --spec <spec>`` (:mod:`yeto.launch.nc.records_relay`).
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

REMOTE_DOCKER = "remote-docker"
AGENTENV = "agentenv"
PROVIDERS = (REMOTE_DOCKER, AGENTENV)
E2B_REQUIREMENT = "e2b==2.64.0"  # the SDK version the AgentENV adapter was verified with
AGENTENV_KEY_FILE_ENV = "AGENTENV_API_KEY_FILE"  # yeto.rl.harness.harbor.agentenv_client.API_KEY_FILE_ENV
PLACEHOLDER = re.compile(r"__TODO|<[^>]*>")
RELAY_UNIT = "yeto-records-relay.service"
RELAY_UNIT_PATH = f"/etc/systemd/system/{RELAY_UNIT}"
RELAY_TAG = "yeto-records-relay"  # the authorized_keys comment of every line records-relay writes
RELAY_CONTAINER = "yeto-records-relay"  # the supervisor container of a docker node without systemd
# "systemd" when PID 1 is systemd and systemctl is there; anything else gets the supervisor
SYSTEMD_PROBE = ('if [ "$(cat /proc/1/comm 2>/dev/null)" = systemd ] && command -v systemctl >/dev/null 2>&1; '
                 'then echo YETO_INIT systemd; else echo YETO_INIT none; fi')

_KEYS = {
    "": {"root", "harbor", "docker_cli", "sandbox", "tunnel", "tasks", "records_secret", "provider", "agentenv",
         "records_proxy", "records_relay"},
    "agentenv": {"gateway_url", "sandbox_url", "api_key", "machine_type", "max_sandboxes", "pool"},
    "records_proxy": {"public_base"},
    "records_relay": {"host", "user", "port", "key", "known_hosts", "listen_port", "public_host", "bind",
                      "allow_cidrs", "settle_s", "ready_timeout_s"},
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
        # the value is not quoted: an inline value here is the secret itself (a key, a token)
        shown = sorted(map(str, value)) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{what} must be {{\"path\": <local file>}} or {{\"env\": <VAR>}}, got {shown} "
                         "(value not shown)")

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
class RelaySection:
    """``agentic.setup.records_relay`` before a shard is picked (the relay side needs
    every shard's port and key: :mod:`yeto.launch.nc.records_relay`)."""

    host: str
    user: str
    port: int
    key: SecretRef  # a path may hold ``{shard}``
    known_hosts: SecretRef | None
    port_lo: int
    port_hi: int
    public_host: str
    bind: str
    allow_cidrs: tuple[str, ...]
    settle_s: float
    ready_timeout_s: float

    @property
    def ports(self) -> range:
        return range(self.port_lo, self.port_hi + 1)

    @property
    def per_shard_key(self) -> bool:
        return bool(self.key.path and "{shard}" in self.key.path)

    def key_of(self, shard: int) -> SecretRef:
        return SecretRef(path=self.key.path.replace("{shard}", str(shard))) if self.per_shard_key else self.key

    def for_shard(self, sampler_id: Any, *, target_port: int) -> RelaySpec:
        if self.port_lo == self.port_hi and not self.per_shard_key:
            shard = None
            listen = self.port_lo
        else:
            tail = str(sampler_id or "").rpartition("-")[2]
            if not tail.isdigit():
                raise ValueError(f"agentic.setup.records_relay: a listen_port range or a {{shard}} key path needs "
                                 f"a sampler_id ending in -<shard index>, got {sampler_id!r}")
            shard = int(tail)
            listen = self.port_lo + shard
            if listen > self.port_hi:
                raise ValueError(f"agentic.setup.records_relay.listen_port {self.port_lo}-{self.port_hi} has no "
                                 f"port for shard {sampler_id!r}: widen it to at least one port per shard")
        return RelaySpec(host=self.host, user=self.user, port=self.port,
                         key=self.key_of(shard) if shard is not None else self.key, known_hosts=self.known_hosts,
                         listen_port=listen, target_port=target_port, public_host=self.public_host, bind=self.bind,
                         settle_s=self.settle_s, ready_timeout_s=self.ready_timeout_s)


@dataclass(frozen=True)
class RelaySpec:
    """One sampler node's reverse forward to the records relay."""

    host: str
    user: str
    port: int
    key: SecretRef
    known_hosts: SecretRef | None
    listen_port: int
    target_port: int
    public_host: str
    bind: str
    settle_s: float
    ready_timeout_s: float

    @property
    def public_base(self) -> str:
        host = f"[{self.public_host}]" if ":" in self.public_host else self.public_host
        return f"http://{host}:{self.listen_port}"

    @property
    def forward(self) -> str:
        return f"{self.bind}:{self.listen_port}:127.0.0.1:{self.target_port}"


@dataclass(frozen=True)
class AgentenvSpec:
    """``agentic.setup.agentenv``: the AgentENV gateway the sampler's Harbor trials dial."""

    gateway_url: str
    sandbox_url: str
    api_key: SecretRef
    machine_type: str | None
    max_sandboxes: int | None
    pool: bool
    records_public_base: str | None  # agentic.setup.records_proxy.public_base (or the relay's)
    relay: RelaySpec | None = None  # agentic.setup.records_relay

    @property
    def records_placeholder(self) -> bool:
        return not self.records_public_base or bool(PLACEHOLDER.search(self.records_public_base))


@dataclass(frozen=True)
class AgenticSetup:
    root: str
    harbor_version: str
    harbor_commit: str
    harbor_keep: tuple[str, ...]
    docker_cli_version: str
    compose_version: str
    sandbox: SandboxAccess | None  # None: provider agentenv (no docker host)
    tunnel: TunnelSpec | None
    tasks: TasksSpec | None
    records_secret: SecretRef | str | None
    notes: tuple[str, ...] = field(default=())
    sandbox_host: str | None = None  # the agentic.envs.hosts name of the sandbox host
    agentenv: AgentenvSpec | None = None

    @property
    def provider(self) -> str:
        return AGENTENV if self.agentenv is not None else REMOTE_DOCKER

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

    @property
    def agentenv_key_file(self) -> str:
        return f"{self.home}/agentenv.key"

    @property
    def relay(self) -> RelaySpec | None:
        return self.agentenv.relay if self.agentenv is not None else None

    def container_env(self, *, python_bin: str = "/opt/runtime-venv/bin") -> dict[str, str]:
        """The sampler container's environment for this setup (an env file ahead of
        the operator's agentic env file). ``PYTHONPATH`` is :meth:`python_path`."""
        env = {"HOME": self.home,
               "PATH": f"{self.root}/bin:{python_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
               "HARBOR_ENV_TYPE": self.provider, "HARBOR_TASKS_DIR": f"{self.root}/tasks",
               "HARBOR_TRIALS_DIR": f"{self.root}/trials"}
        if self.sandbox is not None:
            env["DOCKER_HOST"] = f"ssh://{self.sandbox.alias}"
        if (a := self.agentenv) is not None:
            # the key stays in its 0600 file on the node: never in an env file or a dry-run line
            env.update({"E2B_API_URL": a.gateway_url, "AGENTENV_SANDBOX_URL": a.sandbox_url,
                        "E2B_SANDBOX_URL": a.sandbox_url, AGENTENV_KEY_FILE_ENV: self.agentenv_key_file})
            if a.machine_type:
                env["HARBOR_ENV_KWARGS"] = json.dumps({"machine_type": a.machine_type})
            if a.max_sandboxes:
                env["HARBOR_MAX_SANDBOXES"] = json.dumps({AGENTENV: a.max_sandboxes})
            if a.pool:
                env["YETO_AGENTENV_POOL"] = "1"
            if a.records_public_base:
                env["YETO_RECORDS_PROXY_PUBLIC_BASE"] = a.records_public_base
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

    provider = _provider(doc, agentic)
    if provider == AGENTENV:
        return _parse_agentenv(doc, root=root, version=version, commit=commit, keep=keep,
                               cli_version=cli_version, compose=compose, local=local, ref=ref,
                               spec=spec, agentic=agentic)
    for key in ("agentenv", "records_proxy", "records_relay"):
        if doc.get(key):
            raise ValueError(f"agentic.setup.{key} is for provider agentenv (agentic.setup.provider is "
                             f"{provider!r})")

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

    tasks = _tasks(doc, local)

    secret: SecretRef | str | None = None
    if "records_secret" in doc:
        raw = doc["records_secret"]
        secret = "generate" if raw == "generate" else ref(raw, "agentic.setup.records_secret")
    elif tunnel is not None:
        raise ValueError("agentic.setup: a tunnel makes the records proxy public; set records_secret "
                         "(\"generate\", {\"path\"} or {\"env\"})")

    return AgenticSetup(root=root, harbor_version=version, harbor_commit=commit, harbor_keep=keep,
                        docker_cli_version=cli_version, compose_version=compose, sandbox=sandbox, tunnel=tunnel,
                        tasks=tasks, records_secret=secret, notes=tuple(notes),
                        sandbox_host=None if envs_host is None else envs_host.name)


def _provider(doc: Mapping[str, Any], agentic: Mapping[str, Any]) -> str:
    """``agentic.setup.provider``, else ``task_defaults.provider``, else remote-docker; the two must agree."""
    declared = doc.get("provider")
    defaults = (agentic.get("task_defaults") or {}).get("provider")
    if declared is not None and declared not in PROVIDERS:
        raise ValueError(f"agentic.setup.provider must be one of {list(PROVIDERS)}, got {declared!r}")
    if defaults and defaults not in PROVIDERS:
        raise ValueError(f"agentic.setup prepares {' or '.join(PROVIDERS)} sandboxes; task_defaults.provider "
                         f"is {defaults!r}")
    if declared and defaults and declared != defaults:
        raise ValueError(f"agentic.setup.provider {declared!r} is not task_defaults.provider {defaults!r}")
    return str(declared or defaults or REMOTE_DOCKER)


def _url(value: Any, what: str) -> str:
    text = str(value or "").strip().rstrip("/")
    url = urlsplit(text)
    if url.scheme not in ("http", "https") or not url.hostname or PLACEHOLDER.search(text):
        raise ValueError(f"{what} must be an http(s) URL, got {value!r}")
    return text


def _parse_agentenv(doc: Mapping[str, Any], *, root: str, version: str, commit: str,
                    keep: tuple[str, ...], cli_version: str, compose: str, local: Callable[[str], str],
                    ref: Callable[[Any, str], SecretRef], spec: Mapping[str, Any],
                    agentic: Mapping[str, Any]) -> AgenticSetup:
    """The ``agentic.setup`` of an AgentENV run: no docker host, no reverse tunnel."""
    for key in ("sandbox", "tunnel", "docker_cli"):
        if doc.get(key):
            raise ValueError(f"agentic.setup.{key} is for remote-docker sandboxes; AgentENV microVMs have no "
                             "docker host (drop it with provider agentenv)")
    notes: list[str] = []
    ae = _section(doc, "agentenv")
    gateway = _url(ae.get("gateway_url"), "agentic.setup.agentenv.gateway_url")
    sandbox_url = _url(ae.get("sandbox_url"), "agentic.setup.agentenv.sandbox_url") if ae.get("sandbox_url") \
        else gateway
    if "api_key" not in ae:
        raise ValueError("agentic.setup.agentenv.api_key is required ({\"path\": <env-file>} or {\"env\": <VAR>})")
    key = ref(ae["api_key"], "agentic.setup.agentenv.api_key")
    cap = ae.get("max_sandboxes")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
        raise ValueError(f"agentic.setup.agentenv.max_sandboxes must be a positive integer, got {cap!r}")
    if not isinstance(ae.get("pool", False), bool):
        raise ValueError("agentic.setup.agentenv.pool must be true or false")
    public = _section(doc, "records_proxy").get("public_base")
    proxy_port = (agentic.get("proxy") or {}).get("port")
    relay = None
    if doc.get("records_relay"):
        if public:
            raise ValueError("agentic.setup: give records_proxy.public_base or records_relay, not both (the relay "
                             "sets the public base)")
        if not proxy_port:
            raise ValueError("agentic.setup.records_relay needs a fixed records proxy port: set agentic.proxy.port")
        relay = parse_relay(doc, ref=ref).for_shard(
            spec.get("sampler_id"), target_port=_port(proxy_port, "agentic.proxy.port"))
        public = relay.public_base
        if relay.known_hosts is None:
            notes.append("records_relay.known_hosts is unset: the first connection trusts the relay's host key "
                         "(StrictHostKeyChecking accept-new); pin it for production")
    a = AgentenvSpec(gateway_url=gateway, sandbox_url=sandbox_url, api_key=key,
                     machine_type=str(ae["machine_type"]) if ae.get("machine_type") else None,
                     max_sandboxes=cap, pool=bool(ae.get("pool", False)),
                     records_public_base=str(public).strip().rstrip("/") if public else None, relay=relay)
    if a.records_placeholder:
        notes.append(f"WARNING: agentic.setup.records_proxy.public_base is {public!r}: the microVMs deny private "
                     "egress, so trials cannot reach the records proxy and no trial is judged; set a public "
                     "https base or a records_relay (the node setup refuses to start without one)")
    elif relay is not None:
        notes.append(f"records_relay base {a.records_public_base} is plain http: the records secret travels in "
                     "the URL path in clear text; keep records_relay.allow_cidrs to the sandbox egress addresses")
    else:
        if direct := direct_node_warning(a.records_public_base, proxy_port):
            notes.append(direct)
        if urlsplit(a.records_public_base).scheme != "https":
            notes.append(f"records_proxy.public_base {a.records_public_base} is not https: the records secret "
                         "travels in the URL path in clear text")
    if urlsplit(gateway).scheme != "https":
        notes.append(f"agentenv.gateway_url {gateway} is plain http: the API key travels in clear text")
    tasks = _tasks(doc, local)
    raw = doc.get("records_secret", "generate")
    secret = "generate" if raw == "generate" else ref(raw, "agentic.setup.records_secret")
    return AgenticSetup(root=root, harbor_version=version, harbor_commit=commit, harbor_keep=keep,
                        docker_cli_version=cli_version, compose_version=compose, sandbox=None, tunnel=None,
                        tasks=tasks, records_secret=secret, notes=tuple(notes), agentenv=a)


def parse_relay(doc: Mapping[str, Any], *, ref: Callable[[Any, str], SecretRef] | None = None) -> RelaySection:
    """``agentic.setup.records_relay`` of an ``agentic.setup`` section ``doc``."""
    ref = ref or SecretRef.parse
    rl = _section(doc, "records_relay")
    host = str(rl.get("host") or "").strip()
    if not host or PLACEHOLDER.search(host) or "/" in host or "@" in host:
        raise ValueError(f"agentic.setup.records_relay.host must be the relay's ssh host name or address, "
                         f"got {rl.get('host')!r}")
    user = str(rl.get("user", "root"))
    if not _NAME.match(user):
        raise ValueError(f"agentic.setup.records_relay.user {user!r} is not a user name")
    if "key" not in rl:
        raise ValueError("agentic.setup.records_relay.key is required (this node's forward key: {\"path\"} or "
                         "{\"env\"}; a path may hold {shard} for one key per shard)")
    raw = str(rl.get("listen_port") or "")
    lo_s, dash, hi_s = raw.partition("-")
    if not lo_s:
        raise ValueError("agentic.setup.records_relay.listen_port is required (<port> or <lo>-<hi>: shard k "
                         "listens on lo + k)")
    lo = _port(lo_s, "agentic.setup.records_relay.listen_port")
    hi = _port(hi_s, "agentic.setup.records_relay.listen_port range end") if dash else lo
    if hi < lo:
        raise ValueError(f"agentic.setup.records_relay.listen_port range {raw!r} is empty")
    public_host = str(rl.get("public_host") or host).strip()
    if PLACEHOLDER.search(public_host) or "/" in public_host:
        raise ValueError(f"agentic.setup.records_relay.public_host {public_host!r} must be a host name or address")
    bind = str(rl.get("bind", "0.0.0.0"))
    cidrs = rl.get("allow_cidrs") or []
    if isinstance(cidrs, str) or not isinstance(cidrs, Sequence):
        raise ValueError("agentic.setup.records_relay.allow_cidrs must be a list of CIDRs")
    import ipaddress  # noqa: PLC0415

    allow = []
    for c in cidrs:
        try:
            allow.append(str(ipaddress.ip_network(str(c), strict=False)))
        except ValueError:
            raise ValueError(f"agentic.setup.records_relay.allow_cidrs: {c!r} is not a CIDR") from None
    known = ref(rl["known_hosts"], "agentic.setup.records_relay.known_hosts") if rl.get("known_hosts") else None
    return RelaySection(host=host, user=user, port=_port(rl.get("port", 22), "agentic.setup.records_relay.port"),
                        key=ref(rl["key"], "agentic.setup.records_relay.key"), known_hosts=known, port_lo=lo,
                        port_hi=hi, public_host=public_host, bind=bind, allow_cidrs=tuple(allow),
                        settle_s=float(rl.get("settle_s", 5)), ready_timeout_s=float(rl.get("ready_timeout_s", 60)))


DIRECT_NODE_HINT = ("a sampler node reachable on port 22 only (National Compute: a private address behind a 1:1 "
                    "NAT whose cloud firewall passes 22 alone) never sees the microVMs' SYNs, so every trial "
                    "fails with no recorded model call and a zero reward; put the proxy behind a "
                    "records_relay (docs/NC_POOL.md, AgentENV sandboxes)")


# documentation ranges count as public here (``is_private`` says otherwise): examples and tests use them
_DOC_NETS = ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "2001:db8::/32")
_SHARED_NET = "100.64.0.0/10"  # carrier-grade NAT: not ``is_private``, still internal


def direct_node_warning(public_base: str | None, proxy_port: Any, *, node_ip: str | None = None) -> str | None:
    """A warning when the records public base points straight at a sampler node's
    records proxy: the node's own address (``node_ip``, when known), or a public IP
    literal on ``agentic.proxy.port`` (a dry run, which does not know the node)."""
    if not public_base or PLACEHOLDER.search(public_base):
        return None
    url = urlsplit(public_base)
    host = url.hostname or ""
    if node_ip and host == node_ip:
        return f"WARNING: records public base {public_base} is the sampler node's own address: {DIRECT_NODE_HINT}"
    import ipaddress  # noqa: PLC0415

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    nets = [ipaddress.ip_network(n) for n in (*_DOC_NETS, _SHARED_NET)]
    doc = any(ip.version == net.version and ip in net for net in nets[:-1])
    if ip.is_loopback or ip.is_link_local or (ip.is_private and not doc) or (ip.version == 4 and ip in nets[-1]):
        return None  # an internal address (a private sandbox network): the egress allow-list decides
    try:
        direct = proxy_port is not None and url.port is not None and int(url.port) == int(proxy_port)
    except (TypeError, ValueError):
        direct = False
    if direct:
        return (f"WARNING: records public base {public_base} is an IP on the records proxy port "
                f"{proxy_port} (the sampler node's own address?): {DIRECT_NODE_HINT}")
    return None


def _tasks(doc: Mapping[str, Any], local: Callable[[str], str]) -> TasksSpec | None:
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
    return tasks


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
    s = setup.sandbox
    for what, r, private in (("sandbox.key", s.key if s else None, True),
                             ("sandbox.known_hosts", s.known_hosts if s else None, False),
                             ("tunnel.key", setup.tunnel.key if setup.tunnel else None, True),
                             ("agentenv.api_key", setup.agentenv.api_key if setup.agentenv else None, True),
                             ("records_relay.key", setup.relay.key if setup.relay else None, True),
                             ("records_relay.known_hosts", setup.relay.known_hosts if setup.relay else None, False)):
        if r is not None and (p := r.problem(environ, private=private)):
            out.append(f"agentic.setup.{what}: {p}")
        elif what == "agentenv.api_key" and r is not None:
            try:
                if not agentenv_key_of(r.read(environ)):
                    out.append(f"agentic.setup.agentenv.api_key: {r.describe()} holds no AGENTENV_API_KEY= / "
                               "E2B_API_KEY= line or bare key (value not shown)")
            except (OSError, ValueError) as error:
                out.append(f"agentic.setup.agentenv.api_key: {type(error).__name__} reading {r.describe()}")
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


def agentenv_key_of(text: str) -> str | None:
    """:func:`yeto.rl.harness.harbor.agentenv_client.key_from_text` (imported lazily: the
    launcher need not import the harness package)."""
    from yeto.rl.harness.harbor.agentenv_client import key_from_text  # noqa: PLC0415

    return key_from_text(text)


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
    s, t, a = setup.sandbox, setup.tunnel, setup.agentenv
    lines = [f"agentic setup under {setup.root} (node: <nvme>{setup.root}), provider {setup.provider}"]
    if s is not None:
        lines.append(f"  docker_cli: docker {setup.docker_cli_version} + compose v{setup.compose_version} in "
                     f"{setup.root}/bin")
    lines.append(f"  harbor: {setup.harbor_version} @ {setup.harbor_commit[:12]} into {setup.site} "
                 f"(keep {list(setup.harbor_keep)})" + (f" + {E2B_REQUIREMENT}" if a else ""))
    if s is not None:
        lines.append(f"  sandbox: DOCKER_HOST=ssh://{s.alias} -> {s.user}@{s.host}:{s.port}"
                     + (f" (agentic.envs host {setup.sandbox_host})" if setup.sandbox_host else "")
                     + f", key {s.key.describe()}, "
                     f"known_hosts {s.known_hosts.describe() if s.known_hosts else 'accept-new'}")
    if a is not None:
        lines.append(f"  agentenv: gateway {a.gateway_url}, data plane {a.sandbox_url}, key {a.api_key.describe()} "
                     f"-> {setup.agentenv_key_file} (0600, ${AGENTENV_KEY_FILE_ENV}); no docker host, no tunnel")
        lines.append(f"  records_proxy: public base {a.records_public_base or '-'}")
    if (r := setup.relay) is not None:
        lines.append(f"  records_relay: {RELAY_UNIT} (Restart=always; without systemd the reconnecting supervisor) "
                     f"ssh -N -R {r.forward} {r.user}@{r.host}:{r.port}"
                     f", key {r.key.describe()}, known_hosts {r.known_hosts.describe() if r.known_hosts else 'accept-new'}"
                     f"; the relay side: python -m yeto.launch.nc records-relay --spec <spec>")
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
            if tar.stdout is not None:  # an early ssh exit must not leave tar blocked on our pipe
                tar.stdout.close()
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


def classify_relay_failure(text: str, *, who: str, fingerprint: str, listen: str) -> str:
    """An operator message for a failed reverse forward to the records relay."""
    low = text.lower()
    fix = "run `python -m yeto.launch.nc records-relay --spec <spec>` against the relay (docs/NC_POOL.md, " \
          "AgentENV sandboxes)"
    if "permission denied" in low and "publickey" in low:
        return (f"the records relay refused this node's key ({who}): the public key with fingerprint "
                f"{fingerprint} needs its restricted authorized_keys line there; {fix}")
    if "remote port forwarding failed" in low or "forwarding failed" in low:
        return (f"the records relay refused the reverse forward {listen} ({who}): the key's authorized_keys line "
                f"needs permitlisten for it and sshd needs GatewayPorts clientspecified ({fix}); or another "
                "tunnel holds the port (an older hand-made relay service on this node or another shard's node)")
    return classify_ssh_failure(text, who=who, key="records relay key", fingerprint=fingerprint,
                                purpose=f"records relay forward {listen}")


def relay_ssh_config(r: RelaySpec, relay_dir: str) -> str:
    return "\n".join([
        f"Host {RELAY_TAG}", f"  HostName {r.host}", f"  User {r.user}", f"  Port {r.port}",
        f"  IdentityFile {relay_dir}/key", "  IdentitiesOnly yes", "  BatchMode yes",
        f"  StrictHostKeyChecking {'yes' if r.known_hosts else 'accept-new'}",
        f"  UserKnownHostsFile {relay_dir}/known_hosts", "  ExitOnForwardFailure yes",
        "  ServerAliveInterval 15", "  ServerAliveCountMax 3", "  ConnectTimeout 20", ""])


def relay_unit(r: RelaySpec, relay_dir: str) -> str:
    """The systemd unit of a node's reverse forward: restarted forever (no start limit)."""
    return "\n".join([
        "[Unit]", "Description=yeto records-proxy reverse forward to the records relay",
        "After=network-online.target", "Wants=network-online.target", "StartLimitIntervalSec=0", "",
        "[Service]", f"ExecStart=/usr/bin/ssh -N -F {relay_dir}/config -R {r.forward} {RELAY_TAG}",
        "Restart=always", "RestartSec=5", "", "[Install]", "WantedBy=multi-user.target", ""])


def relay_supervisor_argv(r: RelaySpec, relay_dir: str, python: str) -> list[str]:
    """The forward of :func:`relay_unit` under yeto's reconnecting supervisor (a host
    without systemd): the same ssh command, restarted forever with a backoff of 5 s
    doubling to 60 s (reset after a connection that stayed up), each up / down a line
    in ``<relay_dir>/events.jsonl``."""
    return [python, "-m", "yeto.launch.agentic_node_agent", "tunnel", "--events", f"{relay_dir}/events.jsonl",
            "--settle-s", f"{r.settle_s:g}", "--retry-s", "5", "--max-backoff-s", "60", "--",
            "ssh", "-N", "-F", f"{relay_dir}/config", "-R", r.forward, RELAY_TAG]


# -- env files ---------------------------------------------------------------------

def env_file_text(env: Mapping[str, str], *, style: str) -> str:
    """``KEY=VALUE`` lines of ``env`` (sorted) for one reader:

    ``docker``  ``docker run --env-file``: the value is taken literally (no quotes are
                removed, nothing is expanded), so it is written as is; a newline cannot
                be written and is refused.
    ``shell``   ``set -a; . <file>`` and systemd ``EnvironmentFile=``: every value is
                single-quoted (:func:`shlex.quote`; an inner ``'`` becomes ``'"'"'``), which
                both readers strip and take literally: a JSON value with spaces and quotes
                (``HARBOR_ENV_KWARGS``) is one value, not words and a command."""
    if style not in ("docker", "shell"):
        raise ValueError(f"env file style {style!r}: docker or shell")
    lines = []
    for key, value in sorted(env.items()):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
            raise ValueError(f"env file: {key!r} is not a variable name")
        value = str(value)
        if style == "docker":
            if "\n" in value or "\r" in value:
                raise ValueError(f"env file: the value of {key} has a newline (docker --env-file cannot hold it)")
            lines.append(f"{key}={value}\n")
        else:
            lines.append(f"{key}={shlex.quote(value)}\n")
    return "".join(lines)


def shell_env_file(text: str) -> str:
    """An operator env file written for ``docker run --env-file`` (the docker route's
    reader: ``KEY=VALUE`` literally) made safe to ``source``: an unquoted value is
    single-quoted, so the process route reads what the docker route reads. A value
    that already starts with a quote, an ``export`` line, a comment and anything
    that is not ``KEY=VALUE`` are kept as written (the operator wrote shell)."""
    out = []
    for line in text.splitlines():
        body = line.lstrip()  # docker trims the left only: a trailing space is part of the value
        name, eq, value = body.partition("=")
        if (not eq or body.startswith("#") or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                or value[:1] in ("'", '"')):
            out.append(line)
            continue
        out.append(f"{name}={shlex.quote(value)}")
    return "\n".join(out) + ("\n" if out else "")


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
    for name in sys.argv[1:]:  # the provider's SDK (agentenv: e2b)
        __import__(name)
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
        a = self.setup.agentenv
        if a is not None and a.records_placeholder:
            raise self._fail("records_proxy", f"agentic.setup.records_proxy.public_base is "
                                              f"{a.records_public_base!r}: AgentENV microVMs deny private egress, "
                                              "so no trial could reach the records proxy; set the public https base")
        self.layout()
        if self.setup.sandbox is not None:  # agentenv: no docker host
            self.docker_cli()
        self.harbor()
        if self.setup.sandbox is not None:
            self.sandbox()
        self.records_secret()
        self.tasks()
        self.agentenv_key()
        self.records_relay()
        self.tunnel()
        return dict(self.report)

    @property
    def relay_dir(self) -> str:
        """The records relay key directory on the node (outside the sampler's ``/work``)."""
        return f"{self.node.nvme}/records-relay"

    def _relay_state(self) -> tuple[str, str]:
        out = self.node.sh(f"sudo systemctl is-active {RELAY_UNIT} 2>/dev/null; sudo systemctl show -p NRestarts "
                           f"--value {RELAY_UNIT} 2>/dev/null", timeout=60)
        parts = (out.stdout or "").split()
        return (parts[0] if parts else "unknown"), (parts[1] if len(parts) > 1 else "")

    def _has_systemd(self) -> bool:
        """PID 1 is systemd and ``systemctl`` is there (a Daytona / Modal sandbox and most
        containers have neither: the forward then runs under the supervisor)."""
        out = self.node.sh(SYSTEMD_PROBE, timeout=60)
        return "YETO_INIT systemd" in (out.stdout or "")

    @property
    def _relay_events(self) -> str:
        return f"{self.relay_dir}/events.jsonl"

    def _relay_supervisor_state(self) -> tuple[str, str]:
        """``(state, restarts)`` of the supervised forward in the systemd vocabulary:
        ``active`` once the supervisor runs and its last event is ``up``; ``restarts``
        counts the ``down`` events (the events file is emptied at each supervisor start)."""
        ev = shlex.quote(self._relay_events)
        out = self.node.sh(f"sudo grep -c '\"state\": \"down\"' {ev} 2>/dev/null; sudo tail -n 1 {ev} 2>/dev/null",
                           timeout=60)
        lines = (out.stdout or "").splitlines()
        downs = lines[0].strip() if lines and lines[0].strip().isdigit() else "0"
        last = _events("\n".join(lines[1:]))
        up = bool(last) and last[-1].get("state") == "up"
        return ("active" if up and self._relay_supervisor_running() else "activating"), downs

    def _relay_supervisor_running(self) -> bool:
        out = self.node.sh(f"sudo docker inspect -f '{{{{.State.Running}}}}' {RELAY_CONTAINER} 2>/dev/null || true",
                           timeout=60)
        return (out.stdout or "").strip() == "true"

    def _relay_supervisor_start(self, argv: Sequence[str]) -> Any:
        """A docker node without systemd: the supervisor in a helper container of the
        sampler image (host network, ``--restart unless-stopped``), the relay directory
        mounted at its own path (the ssh config names it)."""
        d = self.relay_dir
        run = ["sudo", "docker", "run", "-d", "--name", RELAY_CONTAINER, "--restart", "unless-stopped",
               "--network", "host", "-v", f"{self.node.yeto_dir}:/yeto:ro", "-v", f"{d}:{d}",
               "-e", "PYTHONPATH=/yeto", "--entrypoint", argv[0], self.node.image, *argv[1:]]
        return self.node.sh(f"sudo docker rm -f {RELAY_CONTAINER} >/dev/null 2>&1; "
                            f"sudo sh -c {shlex.quote(': > ' + shlex.quote(self._relay_events))}; "
                            + shlex.join(run), timeout=300)

    def _relay_log(self, systemd: bool) -> str:
        if systemd:
            return self.node.sh(f"sudo journalctl -u {RELAY_UNIT} -n 30 --no-pager 2>&1", timeout=60).stdout or ""
        out = self.node.sh(f"sudo tail -n 30 {shlex.quote(self._relay_events)} 2>/dev/null", timeout=60)
        return "\n".join(str(e.get("stderr") or "") for e in _events(out.stdout) if e.get("state") == "down")

    def records_relay(self) -> None:
        """The node's reverse forward to the records relay: a systemd unit
        (``Restart=always``), or on a host without systemd the same ssh command under
        yeto's reconnecting supervisor (:func:`relay_supervisor_argv`). The key and ssh
        config sit under :attr:`relay_dir` (0700), never in ``/work``; checked by the
        forward staying up with no restart for ``settle_s``."""
        r = self.setup.relay
        if r is None:
            return
        d = self.relay_dir
        out = self.node.sh(f"sudo mkdir -p {shlex.quote(d)} && sudo chmod 700 {shlex.quote(d)}", timeout=60)
        if out.returncode != 0:
            raise self._fail("records_relay", f"creating {d}: {(out.stderr or '').strip()[-300:]}")
        changed = self.node.put(r.key.read(self.environ), f"{d}/key")
        changed |= self.node.put(relay_ssh_config(r, d), f"{d}/config", mode="644")
        if r.known_hosts:
            changed |= self.node.put(r.known_hosts.read(self.environ), f"{d}/known_hosts", mode="644")
        systemd = self._has_systemd()
        if systemd:
            what, state_of = RELAY_UNIT, self._relay_state
            changed |= self.node.put(relay_unit(r, d), RELAY_UNIT_PATH, mode="644")
            state, _ = state_of()
            if changed or state != "active":
                out = self.node.sh(f"sudo systemctl daemon-reload && sudo systemctl enable {RELAY_UNIT} "
                                   f">/dev/null 2>&1 && sudo systemctl restart {RELAY_UNIT}", timeout=120)
                if out.returncode != 0:
                    raise self._fail("records_relay", f"starting {RELAY_UNIT}: {(out.stderr or '').strip()[-300:]}")
        else:
            what, state_of = "the records relay supervisor (no systemd)", self._relay_supervisor_state
            argv = relay_supervisor_argv(r, d, self.node.python)
            changed |= self.node.put(shlex.join(argv) + "\n", f"{d}/supervisor.argv", mode="644")
            state = "active" if self._relay_supervisor_running() else "inactive"
            if changed or state != "active":
                out = self._relay_supervisor_start(argv)
                if out.returncode != 0:
                    raise self._fail("records_relay", f"starting {what}: {(out.stderr or '').strip()[-300:]}")
        deadline = self.clock() + max(r.ready_timeout_s, r.settle_s)
        before = state_of()
        while True:
            self.sleep(r.settle_s)
            now = state_of()
            if before[0] == now[0] == "active" and before[1] == now[1]:
                return self._done("records_relay", "changed" if changed or state != "active" else "ok",
                                  f"-R {r.forward} via {r.user}@{r.host}:{r.port}; public base {r.public_base}"
                                  + ("" if systemd else "; supervisor (no systemd)"))
            if self.clock() >= deadline:
                fpr, _ = key_identity(r.key, environ=self.environ, run=self.local_run)
                raise self._fail("records_relay", classify_relay_failure(
                    self._relay_log(systemd), who=f"{r.user}@{r.host}:{r.port}", fingerprint=fpr,
                    listen=f"{r.bind}:{r.listen_port}") + f" ({what} {now[0]}, restarts {now[1] or '?'})")
            before = now

    def agentenv_key(self) -> None:
        """The AgentENV API key file in ``<root>/home`` (0600): the sampler reads it through
        ``AGENTENV_API_KEY_FILE``; the key is in no env file, no argv and no event."""
        a = self.setup.agentenv
        if a is None:
            return
        changed = self.node.put(a.api_key.read(self.environ), self.node.host(self.setup.agentenv_key_file))
        self._done("agentenv_key", "changed" if changed else "ok", f"{a.api_key.describe()} -> "
                                                                   f"{self.setup.agentenv_key_file}")

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
        extra = " e2b" if self.setup.agentenv is not None else ""
        out = self.node.in_image(f"{shlex.quote(self.node.python)} -c {shlex.quote(HARBOR_CHECK)}{extra}",
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
        sdk = f" {shlex.quote(E2B_REQUIREMENT)}" if self.setup.agentenv is not None else ""
        out = self.node.in_image(
            f"set -e; {py} -m pip install --quiet --disable-pip-version-check --upgrade --target {site} "
            f"{shlex.quote(req)} {shlex.quote(dash)}{sdk}; PYTHONPATH= {py} -c {shlex.quote(HARBOR_PRUNE)} {site} "
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
