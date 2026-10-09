"""Terminal-Bench-2 ``EnvironmentProvider`` (design R-TB / gap G1).

One sandbox per trajectory, started from the task's **official image**
(``task.toml [environment].docker_image``, the exact image the TB2 harness
itself runs).  The sandbox runs nothing but ``sleep infinity``; every agent
command is relayed from the *trusted* side through the backend's ``exec``.
The worker-facing wire (``/execute`` ``/submit``) is therefore served by
``environment.serve_environment`` **in the trusted process**, and the untrusted
worker only ever sees ``http://127.0.0.1:<port>`` plus a bearer token — exactly
the surface ``HttpTerminalEnvironment`` already speaks.  Compared with the
upstream recipe (``tb2_sandbox_modal`` + an OpenEnv ``tbench2_env`` server
layer baked into every task image) this needs no extra image layers, no
tunnel and no Python inside the task image.

Verification mirrors the official harness's stage-at-verify model: ``tests/``
never enters the sandbox before ``evaluate`` (the agent cannot read the
checks), is staged at ``/tests`` from the trusted side when the episode is
over, ``bash /tests/test.sh`` runs in the task workdir and the reward is the
``/logs/verifier/reward.txt`` the script writes (``1``/``0``).  ``solution/``
is never shipped at all (it exists only for the R-TB positive-path proof, run
by the operator through the same relay).

Backends implement ``SandboxBackend``; ``LocalProcessBackend`` (CPU tests and
dev) runs commands in a scratch directory on this host, ``ModalSandboxBackend``
materialises a Modal Sandbox.  ``module:callable`` factories for
``YETO_HARNESS_ENVIRONMENT_PROVIDER``::

    yeto.rl.harness.codex.tb2_provider:modal_provider
    yeto.rl.harness.codex.tb2_provider:local_provider

Configuration (environment variables, read when the factory runs):

``YETO_HARNESS_TB2_TASKS_DIR``   terminal-bench-2 checkout root (required);
                                 ``task_id`` = task directory name (the
                                 ``metadata.task_id`` make_tbench2_data.py writes)
``YETO_HARNESS_TB2_LEASE_SECONDS`` hard per-trajectory wall clock (default:
                                 the task's ``[agent].timeout_sec``, 900)
``YETO_HARNESS_TB2_MODAL_APP``   Modal app the sandboxes are created under
                                 (default ``yeto-tbench2``; sweep scope)
``YETO_HARNESS_TB2_SANDBOX_TTL_S`` / ``YETO_HARNESS_TB2_IDLE_TIMEOUT_S``
                                 Modal hard ceiling / idle reclamation
``YETO_HARNESS_TB2_NETWORK_POLICY`` path of a JSON network policy that
                                 replaces the built-in one (see
                                 ``NetworkPolicy``); sandbox egress is closed
                                 for every task the policy does not grant
``YETO_HARNESS_TB2_FAULT``       fault injection for the R-TB coverage matrix,
                                 comma separated ``kind:selector[=value]``:
                                 ``create_fail:<sel>`` acquire raises
                                 (-> INFRASTRUCTURE / ABORTED);
                                 ``deadline:<sel>=<s>`` lease deadline of
                                 ``s`` seconds (wall-clock cancel -> timeout,
                                 destroy, env_live back to 0);
                                 ``max_turns:<sel>=<n>`` worker gets
                                 ``SECRLENV_MAX_TURNS=n`` (turn budget).
                                 ``<sel>`` is the 0-based acquire ordinal in
                                 this process or an exact trajectory id.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from .client import EpisodeAPIError
from .environment import serve_environment

TASKS_DIR_ENV = "YETO_HARNESS_TB2_TASKS_DIR"
LEASE_SECONDS_ENV = "YETO_HARNESS_TB2_LEASE_SECONDS"
FAULT_ENV = "YETO_HARNESS_TB2_FAULT"
MODAL_APP_ENV = "YETO_HARNESS_TB2_MODAL_APP"
SANDBOX_TTL_ENV = "YETO_HARNESS_TB2_SANDBOX_TTL_S"
IDLE_TIMEOUT_ENV = "YETO_HARNESS_TB2_IDLE_TIMEOUT_S"
ENV_HOST_ENV = "YETO_HARNESS_TB2_ENV_HOST"
NETWORK_POLICY_ENV = "YETO_HARNESS_TB2_NETWORK_POLICY"

DEFAULT_MODAL_APP = "yeto-tbench2"
DEFAULT_SANDBOX_TTL_S = 1800
DEFAULT_IDLE_TIMEOUT_S = 600
DEFAULT_WORKDIR = "/app"
TESTS_PATH = "/tests"
VERIFIER_LOGS_PATH = "/logs/verifier"
MAX_VERIFIER_OUTPUT_BYTES = 64 * 1024
FAULT_KINDS = ("create_fail", "deadline", "max_turns")
TURN_BUDGET_ENV = "SECRLENV_MAX_TURNS"


# ----------------------------------------------------------------------------- task


@dataclass(frozen=True)
class Tb2Task:
    task_id: str
    task_dir: Path
    docker_image: str
    cpus: int
    memory_mb: int
    workdir: str
    agent_timeout_s: float
    verifier_timeout_s: float
    # Task statement from ``instruction.md`` (the Terminal-Bench prompt).  The
    # smoke datasets carry only ``task_id`` and a generic system message, so
    # this is the only place the model can learn what to do (S15 stage 2: all
    # 48 trajectories ran without it and scored 0).
    instruction: str | None = None

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    @property
    def solution_dir(self) -> Path:
        return self.task_dir / "solution"


_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def resolve_task(task_ref: str, tasks_dir: Path) -> Tb2Task:
    """``task_ref`` is the task directory name (``metadata.task_id``); fail closed otherwise."""
    if not isinstance(task_ref, str) or not _TASK_ID_RE.match(task_ref):
        raise ValueError(f"invalid TB2 task reference {task_ref!r}")
    task_dir = Path(tasks_dir) / task_ref
    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        raise FileNotFoundError(f"{task_ref}: no task.toml under {tasks_dir}")
    config = tomllib.loads(toml_path.read_text())
    env_cfg = config.get("environment", {})
    image = env_cfg.get("docker_image")
    if not image:
        raise ValueError(f"{task_ref}: task.toml has no [environment].docker_image")
    if not (task_dir / "tests" / "test.sh").is_file():
        raise FileNotFoundError(f"{task_ref}: tests/test.sh is missing")
    return Tb2Task(
        task_id=task_ref,
        task_dir=task_dir,
        docker_image=str(image),
        cpus=max(1, int(env_cfg.get("cpus", 1))),
        memory_mb=max(2048, int(env_cfg.get("memory_mb", 2048))),
        workdir=_dockerfile_workdir(task_dir / "environment" / "Dockerfile"),
        agent_timeout_s=float(config.get("agent", {}).get("timeout_sec", 900.0)),
        verifier_timeout_s=float(config.get("verifier", {}).get("timeout_sec", 900.0)),
        instruction=_read_instruction(task_dir / "instruction.md"),
    )


def tb2_instructions_env() -> dict[str, str]:
    """Worker env selecting the signed TB2 Codex system prompt."""
    from . import codex_harness_agent as harness

    return {
        harness.INSTRUCTIONS_FAMILY_ENV: harness.TB2_INSTRUCTIONS_FAMILY,
        harness.TB2_INSTRUCTIONS_SHA_ENV: harness.TB2_BASE_INSTRUCTIONS_SHA256,
    }


class TaskPromptPreflightError(ValueError):
    """A dataset row would reach Codex without a real task statement."""


def preflight_task_prompts(data_path: Path, tasks_dir: Path) -> list[tuple[str, str]]:
    """Resolve the first user message of every dataset row exactly as the
    subprocess agent will (``task_prompt``) and refuse rows without a task
    statement or whose prompt is a stringified chat list (S15 stage 2 root
    cause).  Returns ``[(task_id, prompt), ...]`` in row order."""
    import json
    from types import SimpleNamespace

    from .codex_openenv_subprocess_agent_function import TaskPromptMissing, task_prompt

    out: list[tuple[str, str]] = []
    for lineno, line in enumerate(Path(data_path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = dict(row.get("metadata") or {})
        task_id = metadata.get("task_id")
        try:
            task = resolve_task(task_id, Path(tasks_dir))
            prompt = task_prompt(metadata, row.get("prompt"), SimpleNamespace(task=task))
        except (TaskPromptMissing, ValueError, FileNotFoundError) as exc:
            raise TaskPromptPreflightError(f"{data_path}:{lineno} task {task_id!r}: {exc}") from exc
        head = prompt.lstrip()[:2]
        if head in ("[{", "{'", '{"') or "'role':" in prompt or '"role":' in prompt:
            raise TaskPromptPreflightError(f"{data_path}:{lineno} task {task_id!r}: prompt is a stringified chat message")
        out.append((str(task_id), prompt))
    if not out:
        raise TaskPromptPreflightError(f"{data_path}: no dataset rows")
    return out


def _read_instruction(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace").strip()
    return text or None


def _dockerfile_workdir(dockerfile: Path) -> str:
    workdir = DEFAULT_WORKDIR
    if dockerfile.is_file():
        for line in dockerfile.read_text(errors="replace").splitlines():
            m = re.match(r"^\s*WORKDIR\s+(\S+)", line)
            if m:
                workdir = m.group(1).strip('"')
    return workdir


def tests_tar_b64(tests_dir: Path) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(tests_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(tests_dir)))
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------- network policy


@dataclass(frozen=True)
class NetworkGrant:
    """Outbound network access of one sandbox.

    The default grant is closed (no egress).  ``open`` allows all egress.
    Otherwise only ``domains`` (``*.`` wildcard prefixes allowed) and
    ``cidrs`` are reachable.
    """

    open: bool = False
    domains: tuple[str, ...] = ()
    cidrs: tuple[str, ...] = ()

    @property
    def closed(self) -> bool:
        return not self.open and not self.domains and not self.cidrs

    @classmethod
    def parse(cls, value: Any, *, where: str) -> "NetworkGrant":
        if value == "open":
            return cls(open=True)
        if value in ("closed", None):
            return cls()
        if not isinstance(value, dict) or set(value) - {"domains", "cidrs"}:
            raise ValueError(f"{where}: grant must be 'open', 'closed' or {{domains, cidrs}}, got {value!r}")
        domains = tuple(value.get("domains") or ())
        cidrs = tuple(value.get("cidrs") or ())
        for item in domains + cidrs:
            if not isinstance(item, str) or not item or item == "*":
                raise ValueError(f"{where}: bad allowlist entry {item!r} (use 'open' for all egress)")
        import ipaddress

        for cidr in cidrs:
            ipaddress.ip_network(cidr, strict=False)
        return cls(domains=domains, cidrs=cidrs)

    def modal_kwargs(self) -> dict[str, Any]:
        """``modal.Sandbox.create`` network arguments (modal >= 1.5.5)."""
        if self.open:
            return {}
        if self.closed:
            return {"block_network": True}
        kwargs: dict[str, Any] = {}
        if self.cidrs:
            kwargs["outbound_cidr_allowlist"] = list(self.cidrs)
        if self.domains:
            kwargs["outbound_domain_allowlist"] = list(self.domains)
        return kwargs


CLOSED = NetworkGrant()


@dataclass(frozen=True)
class NetworkPolicy:
    """Per-task network grants; every task not listed gets ``default`` (closed).

    JSON form (``YETO_HARNESS_TB2_NETWORK_POLICY``)::

        {"default": "closed",
         "tasks": {"pip-task": {"domains": ["pypi.org", "files.pythonhosted.org"]},
                   "web-task": "open"}}
    """

    tasks: dict[str, NetworkGrant] = field(default_factory=dict)
    default: NetworkGrant = CLOSED

    def grant(self, task_id: str) -> NetworkGrant:
        return self.tasks.get(task_id, self.default)

    @classmethod
    def from_mapping(cls, data: Any, *, where: str = "network policy") -> "NetworkPolicy":
        if not isinstance(data, dict) or set(data) - {"default", "tasks"}:
            raise ValueError(f"{where}: expected {{'default', 'tasks'}}")
        tasks = data.get("tasks", {})
        if not isinstance(tasks, dict):
            raise ValueError(f"{where}: 'tasks' must be an object")
        return cls(
            tasks={str(k): NetworkGrant.parse(v, where=f"{where} task {k}") for k, v in tasks.items()},
            default=NetworkGrant.parse(data.get("default", "closed"), where=f"{where} default"),
        )

    @classmethod
    def builtin(cls) -> "NetworkPolicy":
        from .tb2_network_grants import TB2_OPEN_EGRESS_TASKS

        return cls(tasks={t: NetworkGrant(open=True) for t in TB2_OPEN_EGRESS_TASKS})

    @classmethod
    def from_env(cls) -> "NetworkPolicy":
        path = os.environ.get(NETWORK_POLICY_ENV)
        if not path:
            return cls.builtin()
        import json

        return cls.from_mapping(json.loads(Path(path).read_text()), where=path)


# ----------------------------------------------------------------------------- backend protocol


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    output: str
    timed_out: bool = False


class SandboxHandle(Protocol):
    """One running sandbox. ``exec`` is synchronous (run via ``asyncio.to_thread``)."""

    workdir: str

    def exec(self, command: str, *, timeout_s: float, workdir: str | None = None) -> ExecResult: ...

    def terminate(self) -> None: ...

    def alive(self) -> bool: ...


class SandboxBackend(Protocol):
    def create(self, task: Tb2Task, trajectory_id: str) -> SandboxHandle: ...


def shell_with_limits(command: str, *, timeout_s: float, output_bytes: int) -> str:
    """Wrap an agent command: wall clock via coreutils ``timeout``, output capped.

    Exit code 124 (``timeout``) / 137 (KILL after grace) mean the clock expired;
    stdout and stderr are merged and capped at ``output_bytes + 1`` so the relay
    can tell truncation from an exact fit without ever reading more.
    """
    seconds = max(1, int(timeout_s))
    return (
        f"timeout -k 2 {seconds} bash -c {shlex.quote(command)} 2>&1 | head -c {int(output_bytes) + 1}; "
        "exit ${PIPESTATUS[0]}"
    )


_TIMEOUT_EXIT_CODES = frozenset({124, 137})


# ----------------------------------------------------------------------------- local backend


# Only these parent variables reach a local sandbox command (never tokens).
LOCAL_ENV_PASSTHROUGH = ("PATH", "LANG", "LC_ALL", "TZ")
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def minimal_env(parent: dict[str, str] | None = None) -> dict[str, str]:
    """The base environment of a local sandbox command.

    The parent environment is not inherited: it holds Modal / HF / W&B
    tokens and the reward HMAC key.  Only ``LOCAL_ENV_PASSTHROUGH`` is
    copied; ``PATH`` falls back to a standard value.
    """
    parent = os.environ if parent is None else parent
    env = {k: parent[k] for k in LOCAL_ENV_PASSTHROUGH if parent.get(k)}
    env.setdefault("PATH", DEFAULT_PATH)
    env.setdefault("LANG", "C.UTF-8")
    return env


class LocalProcessSandbox:
    """Commands run on this host in a scratch root; ``/tests`` and ``/logs`` are
    redirected into that root through ``TB2_TESTS_DIR`` / ``TB2_VERIFIER_LOGS_DIR``.

    Commands get ``minimal_env`` only.  The network grant is recorded but NOT
    enforced: this backend is for CPU tests and dev on a trusted host.
    """

    def __init__(self, root: Path, workdir: str, network: NetworkGrant = CLOSED) -> None:
        self.root = root
        self.workdir = workdir
        self.network = network
        (root / workdir.lstrip("/")).mkdir(parents=True, exist_ok=True)
        self._alive = True
        self._processes: set[subprocess.Popen] = set()
        self._lock = threading.Lock()

    def host_path(self, sandbox_path: str) -> Path:
        return self.root / sandbox_path.lstrip("/")

    def exec(self, command: str, *, timeout_s: float, workdir: str | None = None) -> ExecResult:
        if not self._alive:
            raise RuntimeError("sandbox is gone")
        cwd = self.host_path(workdir or self.workdir)
        cwd.mkdir(parents=True, exist_ok=True)
        env = {
            **minimal_env(),
            "TB2_TESTS_DIR": str(self.host_path(TESTS_PATH)),
            "TB2_VERIFIER_LOGS_DIR": str(self.host_path(VERIFIER_LOGS_PATH)),
            "HOME": str(self.root),
        }
        proc = subprocess.Popen(
            ["bash", "-lc", command], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with self._lock:
            self._processes.add(proc)
        try:
            try:
                out, _ = proc.communicate(timeout=timeout_s + 5)
            except subprocess.TimeoutExpired:
                self._kill(proc)
                out, _ = proc.communicate()
                return ExecResult(exit_code=124, output=(out or b"").decode("utf-8", "replace"), timed_out=True)
        finally:
            with self._lock:
                self._processes.discard(proc)
        output = (out or b"").decode("utf-8", "replace")
        return ExecResult(exit_code=proc.returncode, output=output, timed_out=proc.returncode in _TIMEOUT_EXIT_CODES)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def terminate(self) -> None:
        """Like a sandbox kill: every command still running dies with the sandbox."""
        self._alive = False
        with self._lock:
            running = list(self._processes)
        for proc in running:
            self._kill(proc)
        shutil.rmtree(self.root, ignore_errors=True)

    def alive(self) -> bool:
        return self._alive and self.root.exists()


class LocalProcessBackend:
    """CPU backend: a scratch directory per sandbox, commands run as subprocesses.

    ``setup`` (optional) seeds the workdir from the task (tests use it to put
    the fixture the verifier checks into place).
    """

    def __init__(self, base_dir: Path | None = None, *, setup: Callable[[Tb2Task, LocalProcessSandbox], None] | None = None,
                 network_policy: NetworkPolicy | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(tempfile.gettempdir()) / "yeto-tb2-local"
        self.network_policy = network_policy or NetworkPolicy()
        self.setup = setup
        self.created: list[LocalProcessSandbox] = []

    def create(self, task: Tb2Task, trajectory_id: str) -> LocalProcessSandbox:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"{task.task_id}-", dir=self.base_dir))
        sandbox = LocalProcessSandbox(root, task.workdir, self.network_policy.grant(task.task_id))
        if self.setup is not None:
            self.setup(task, sandbox)
        self.created.append(sandbox)
        return sandbox


# ----------------------------------------------------------------------------- Modal backend


def _decode_output(data: bytes | str | None) -> str:
    if not data:
        return ""
    return data if isinstance(data, str) else data.decode("utf-8", errors="replace")


class ModalSandbox:
    def __init__(self, sandbox: Any, workdir: str) -> None:
        self._sandbox = sandbox
        self.workdir = workdir
        self.object_id = getattr(sandbox, "object_id", "")

    def exec(self, command: str, *, timeout_s: float, workdir: str | None = None) -> ExecResult:
        # Bytes, decoded leniently: command output cut by the byte limit (or plain
        # binary output) can end mid UTF-8 sequence, and Modal's text mode then
        # raises UnicodeDecodeError inside the relay (S17 M1, 5 tool calls failed).
        process = self._sandbox.exec(
            "bash", "-lc", command, workdir=workdir or self.workdir, timeout=int(timeout_s) + 30, text=False
        )
        output = _decode_output(process.stdout.read())
        err = _decode_output(process.stderr.read())
        code = process.wait()
        return ExecResult(exit_code=int(code), output=output + err, timed_out=code in _TIMEOUT_EXIT_CODES)

    def terminate(self) -> None:
        try:
            self._sandbox.terminate()
        finally:
            detach = getattr(self._sandbox, "detach", None)
            if callable(detach):
                detach()

    def alive(self) -> bool:
        try:
            return self._sandbox.poll() is None
        except Exception:  # noqa: BLE001 - a sandbox the API no longer knows is gone
            return False


SANDBOX_MODAL_TOKEN_ID_ENV = "YETO_SANDBOX_MODAL_TOKEN_ID"
SANDBOX_MODAL_TOKEN_SECRET_ENV = "YETO_SANDBOX_MODAL_TOKEN_SECRET"


def sandbox_modal_client(environ=None) -> Any:
    """secret-handling-hardening: a Modal client built from the sandbox-only
    token (YETO_SANDBOX_MODAL_TOKEN_*), or None when it is not set (then
    Modal's default credentials apply, e.g. on a developer machine)."""
    environ = os.environ if environ is None else environ
    token_id = environ.get(SANDBOX_MODAL_TOKEN_ID_ENV)
    token_secret = environ.get(SANDBOX_MODAL_TOKEN_SECRET_ENV)
    if not (token_id and token_secret):
        return None
    from modal import Client

    return Client.from_credentials(token_id, token_secret)


class ModalSandboxBackend:
    """Modal Sandbox from the task's official image. All ``modal`` imports are lazy."""

    def __init__(
        self,
        *,
        app_name: str = DEFAULT_MODAL_APP,
        ttl_s: int = DEFAULT_SANDBOX_TTL_S,
        idle_timeout_s: int = DEFAULT_IDLE_TIMEOUT_S,
        run_id: str | None = None,
        network_policy: NetworkPolicy | None = None,
    ) -> None:
        self.app_name = app_name
        # Closed for every task unless a policy grants it (task 5.1).
        self.network_policy = network_policy or NetworkPolicy()
        self.ttl_s = int(ttl_s)
        self.idle_timeout_s = int(idle_timeout_s)
        self.run_id = run_id or os.environ.get("OPENENV_RUN_ID") or ""
        self._app: Any = None
        self._client: Any = None
        self._lock = threading.Lock()

    def _client_kwargs(self) -> dict[str, Any]:
        """``{"client": c}`` for the sandbox-only token, ``{}`` otherwise."""
        with self._lock:
            if self._client is None:
                self._client = sandbox_modal_client() or False
            return {"client": self._client} if self._client else {}

    def _get_app(self) -> Any:
        kwargs = self._client_kwargs()
        with self._lock:
            if self._app is None:
                from modal import App

                self._app = App.lookup(self.app_name, create_if_missing=True, **kwargs)
            return self._app

    def create(self, task: Tb2Task, trajectory_id: str) -> ModalSandbox:
        from modal import Image, Sandbox

        tags = {"yeto-tb2-task": task.task_id, "yeto-trajectory": trajectory_id[:63]}
        if self.run_id:
            tags["yeto-run-id"] = self.run_id[:63]
        sandbox = Sandbox.create(
            "sleep",
            "infinity",
            app=self._get_app(),
            image=Image.from_registry(task.docker_image),
            timeout=self.ttl_s,
            idle_timeout=self.idle_timeout_s,
            cpu=float(task.cpus),
            memory=int(task.memory_mb),
            workdir=task.workdir,
            tags=tags,
            **self._client_kwargs(),
            **self.network_policy.grant(task.task_id).modal_kwargs(),
        )
        return ModalSandbox(sandbox, task.workdir)


# ----------------------------------------------------------------------------- relay environment + verifier


class EpisodeBinding:
    """The one episode a sandbox serves: bound by the first request on the wire.

    The agent function mints the episode id after ``acquire`` (the worker job
    carries it), so the lease cannot know it up front; it binds to the first id
    it sees and refuses every other one (one sandbox == one episode).
    """

    def __init__(self) -> None:
        self.episode_id: str | None = None

    def bind(self, episode_id: str) -> None:
        if self.episode_id is None:
            self.episode_id = episode_id
        elif episode_id != self.episode_id:
            raise EpisodeAPIError(404, "unknown_episode", "unknown episode")


class RelayTerminalEnvironment:
    """``TerminalEnvironment`` served in the trusted process; commands go to the sandbox."""

    def __init__(self, handle: SandboxHandle, binding: EpisodeBinding | None = None) -> None:
        self._handle = handle
        self.binding = binding or EpisodeBinding()
        self.commands: list[str] = []
        self.submissions: list[dict[str, Any]] = []
        self.submitted = False

    def _check_episode(self, episode_id: str) -> None:
        self.binding.bind(episode_id)
        if self.submitted:
            raise EpisodeAPIError(409, "episode_terminal", "episode already submitted")

    async def execute(self, episode_id: str, command: str, timeout_seconds: float, output_bytes: int) -> dict[str, Any]:
        self._check_episode(episode_id)
        self.commands.append(command)
        result = await asyncio.to_thread(
            self._handle.exec,
            shell_with_limits(command, timeout_s=timeout_seconds, output_bytes=output_bytes),
            timeout_s=timeout_seconds,
        )
        truncated = len(result.output.encode("utf-8")) > output_bytes
        return {
            "exit_code": result.exit_code,
            "output": result.output[:output_bytes],
            "timed_out": result.timed_out,
            "truncated": truncated,
        }

    async def submit(self, episode_id: str, submission: dict[str, Any]) -> dict[str, Any]:
        self._check_episode(episode_id)
        self.submissions.append(dict(submission))
        self.submitted = True
        return {"accepted": True}


# S17 G3: Modal rejects an exec whose argv exceeds 64 KiB (ARG_MAX), and 7 of
# the 89 TB2 tasks ship tests/ larger than that once tar+base64'd.  Small tests
# stay inline (command unchanged); larger ones are first appended to a staging
# file in the sandbox in chunks (``verifier_stage_commands``), and the verifier
# command decodes that file instead.
MAX_INLINE_TESTS_B64 = 48_000
TESTS_STAGE_CHUNK = 48_000
TESTS_STAGE_PATH = "/tmp/yeto-tb2-tests.tgz.b64"


def verifier_stage_commands(task: Tb2Task) -> list[str]:
    """Commands to run (in order) before ``verifier_command``; empty when inline."""
    b64 = tests_tar_b64(task.tests_dir)
    if len(b64) <= MAX_INLINE_TESTS_B64:
        return []
    commands = [f": > {TESTS_STAGE_PATH}"]
    commands += [f"printf %s {b64[i:i + TESTS_STAGE_CHUNK]} >> {TESTS_STAGE_PATH}"
                 for i in range(0, len(b64), TESTS_STAGE_CHUNK)]
    return commands


def verifier_command(task: Tb2Task) -> str:
    """Stage ``tests/`` at ``/tests``, run ``test.sh`` in the workdir, print the reward.

    The reward line is tagged so it survives arbitrary test output in front of it.
    Large ``tests/`` must be staged first with ``verifier_stage_commands``.
    """
    b64 = tests_tar_b64(task.tests_dir)
    source = (f"echo {b64} | base64 -d" if len(b64) <= MAX_INLINE_TESTS_B64
              else f"base64 -d {TESTS_STAGE_PATH}")
    return (
        f"T=${{TB2_TESTS_DIR:-{TESTS_PATH}}}; L=${{TB2_VERIFIER_LOGS_DIR:-{VERIFIER_LOGS_PATH}}}; "
        f"rm -rf \"$T\" && mkdir -p \"$T\" \"$L\" && {source} | tar xzf - -C \"$T\" && "
        f"bash \"$T/test.sh\"; rc=$?; echo; echo \"YETO_TB2_TESTSH_RC=$rc\"; "
        f"echo \"YETO_TB2_REWARD=$(cat \"$L/reward.txt\" 2>/dev/null | tr -d '[:space:]')\""
    )


def stage_tests(handle: "SandboxHandle", task: Tb2Task) -> None:
    """Run ``verifier_stage_commands`` in ``handle``; raise if any chunk fails."""
    for command in verifier_stage_commands(task):
        result = handle.exec(command, timeout_s=120)
        if result.exit_code != 0:
            raise RuntimeError(f"staging tests/ failed (rc={result.exit_code}): {result.output[-200:]}")


_REWARD_RE = re.compile(r"YETO_TB2_REWARD=(\S*)")
_RC_RE = re.compile(r"YETO_TB2_TESTSH_RC=(\d+)")


VERIFIER_LOG_CHARS = 2000


def verifier_log_excerpt(output: str, limit: int = VERIFIER_LOG_CHARS) -> str:
    """Last ``limit`` chars of the verifier output (pytest prints its verdict last)."""
    text = output if isinstance(output, str) else ""
    return text if len(text) <= limit else "…" + text[-(limit - 1):]


class Tb2Verifier:
    """Trusted-side verifier: ``tests/test.sh`` inside the sandbox, reward from ``reward.txt``."""

    def __init__(self, handle: SandboxHandle, task: Tb2Task, binding: EpisodeBinding | None = None) -> None:
        self._handle = handle
        self._task = task
        self.binding = binding or EpisodeBinding()
        self.evaluations = 0
        self.last_output = ""

    async def evaluate(self, episode_id: str) -> dict[str, Any]:
        bound = self.binding.episode_id
        if bound is not None and episode_id != bound:
            raise ValueError(f"verifier bound to {bound!r}, asked for {episode_id!r}")
        self.evaluations += 1
        await asyncio.to_thread(stage_tests, self._handle, self._task)
        result = await asyncio.to_thread(
            self._handle.exec, verifier_command(self._task), timeout_s=self._task.verifier_timeout_s
        )
        self.last_output = result.output[-MAX_VERIFIER_OUTPUT_BYTES:]
        reward = _REWARD_RE.search(result.output)
        rc = _RC_RE.search(result.output)
        testsh_rc = int(rc.group(1)) if rc else result.exit_code
        return {
            "passed": bool(reward) and reward.group(1) == "1",
            "testsh_rc": testsh_rc,
            "timed_out": result.timed_out,
            # Observe only (S15 verifier probe): the tail of test.sh output
            # (pytest summary / install errors); never part of the signed outcome.
            "log": verifier_log_excerpt(result.output),
        }


# ----------------------------------------------------------------------------- faults


@dataclass(frozen=True)
class FaultSpec:
    kind: str
    selector: str
    value: str | None = None

    def matches(self, ordinal: int, trajectory_id: str) -> bool:
        return self.selector == str(ordinal) or self.selector == trajectory_id


def parse_faults(spec: str | None) -> tuple[FaultSpec, ...]:
    faults: list[FaultSpec] = []
    for raw in (spec or "").split(","):
        item = raw.strip()
        if not item:
            continue
        kind, sep, rest = item.partition(":")
        if not sep or kind not in FAULT_KINDS:
            raise ValueError(f"{FAULT_ENV}: invalid fault {item!r} (kinds: {', '.join(FAULT_KINDS)})")
        selector, _sep, value = rest.partition("=")
        if not selector:
            raise ValueError(f"{FAULT_ENV}: fault {item!r} needs a selector (acquire ordinal or trajectory id)")
        faults.append(FaultSpec(kind, selector, value or None))
    return tuple(faults)


class InjectedCreateFailure(RuntimeError):
    """Raised by ``acquire`` when ``create_fail`` selects the trajectory."""

    injected_fault = True  # not a provider outage (subprocess agent fail-fast counter)


# ----------------------------------------------------------------------------- provider


@dataclass
class Tb2Lease:
    """Returned by ``acquire``: field names match ``subprocess_agent.EnvironmentLease``."""

    env_url: str
    env_token: str
    verifier: Tb2Verifier
    destroy: Callable[[], Any]
    describe: Callable[[], Any]
    deadline_seconds: float
    worker_env: dict[str, str] | None = None
    task: Tb2Task | None = None
    environment: RelayTerminalEnvironment | None = None
    handle: Any = None


class Tb2EnvironmentProvider:
    def __init__(
        self,
        backend: SandboxBackend,
        tasks_dir: Path,
        *,
        lease_seconds: float | None = None,
        faults: tuple[FaultSpec, ...] = (),
        host: str = "127.0.0.1",
    ) -> None:
        self.backend = backend
        self.tasks_dir = Path(tasks_dir)
        self.lease_seconds = lease_seconds
        self.faults = tuple(faults)
        self.host = host
        self.acquired = 0
        self.destroyed = 0
        self.live: dict[str, Tb2Lease] = {}
        self.fault_log: list[tuple[str, str, int]] = []

    def _faults_for(self, ordinal: int, trajectory_id: str) -> dict[str, str | None]:
        hits = {f.kind: f.value for f in self.faults if f.matches(ordinal, trajectory_id)}
        for kind in hits:
            self.fault_log.append((kind, trajectory_id, ordinal))
        return hits

    async def acquire(self, task_id: str, trajectory_id: str) -> Tb2Lease:
        ordinal = self.acquired
        self.acquired += 1
        task = resolve_task(task_id, self.tasks_dir)
        faults = self._faults_for(ordinal, trajectory_id)
        if "create_fail" in faults:
            raise InjectedCreateFailure(f"injected sandbox creation failure for {trajectory_id!r} (acquire #{ordinal})")
        handle = await asyncio.to_thread(self.backend.create, task, trajectory_id)
        token = secrets.token_urlsafe(24)
        binding = EpisodeBinding()
        relay = RelayTerminalEnvironment(handle, binding)
        try:
            runner, url = await serve_environment(relay, token, host=self.host)
        except BaseException:
            await asyncio.to_thread(handle.terminate)
            raise
        deadline = self.lease_seconds if self.lease_seconds is not None else task.agent_timeout_s
        # 2026-10-07: TB2 episodes run under the signed TB2 system prompt.
        worker_env: dict[str, str] | None = tb2_instructions_env()
        if "deadline" in faults:
            deadline = float(faults["deadline"] or 1.0)
        if "max_turns" in faults:
            worker_env = {**(worker_env or {}), TURN_BUDGET_ENV: str(int(faults["max_turns"] or 2))}
        state = {"destroyed": False}

        async def destroy() -> None:
            if state["destroyed"]:
                return
            state["destroyed"] = True
            try:
                await runner.cleanup()
            finally:
                self.live.pop(trajectory_id, None)
                self.destroyed += 1
                await asyncio.to_thread(handle.terminate)

        async def describe() -> str:
            if not state["destroyed"]:
                return "live"
            # Modal reports the exit asynchronously; a short bounded wait keeps
            # ``describe`` honest without stalling the rollout.
            for _ in range(20):
                if not await asyncio.to_thread(handle.alive):
                    return "gone"
                await asyncio.sleep(0.25)
            return "live"

        lease = Tb2Lease(
            env_url=url,
            env_token=token,
            verifier=Tb2Verifier(handle, task, binding),
            destroy=destroy,
            describe=describe,
            deadline_seconds=float(deadline),
            worker_env=worker_env,
            task=task,
            environment=relay,
            handle=handle,
        )
        self.live[trajectory_id] = lease
        return lease


# ----------------------------------------------------------------------------- factories (module:callable)


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _tasks_dir_from_env() -> Path:
    raw = os.environ.get(TASKS_DIR_ENV)
    if not raw:
        raise RuntimeError(f"{TASKS_DIR_ENV} is not set (terminal-bench-2 checkout root)")
    tasks_dir = Path(raw).expanduser().resolve()
    if not tasks_dir.is_dir():
        raise RuntimeError(f"{TASKS_DIR_ENV}={raw}: not a directory")
    return tasks_dir


def _provider(backend: SandboxBackend) -> Tb2EnvironmentProvider:
    return Tb2EnvironmentProvider(
        backend,
        _tasks_dir_from_env(),
        lease_seconds=_env_float(LEASE_SECONDS_ENV, None),
        faults=parse_faults(os.environ.get(FAULT_ENV)),
        host=os.environ.get(ENV_HOST_ENV, "127.0.0.1"),
    )


def require_modal_client() -> None:
    """The Modal client must import in *this* interpreter (the island's python).

    The ports image does not ship it; Modal mounts its own ``modal`` package
    into Function containers without its dependencies, so a bare ``import
    modal`` there fails only at first Sandbox use, inside every rollout
    worker (A-T3-5, codex-smoke-20261003-7).  The launcher installs the client
    in the island setup (``MODAL_CLIENT_SETUP``); this check makes the driver
    preflight fail closed before any GPU work when that did not happen.
    """
    try:
        import modal  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - ImportError or Modal's own re-raise
        raise RuntimeError(
            f"the Modal Sandbox backend needs an importable `modal` client in {sys.executable}: {exc}"
        ) from exc


def modal_provider(miles_args: Any = None) -> Tb2EnvironmentProvider:
    """``YETO_HARNESS_ENVIRONMENT_PROVIDER=yeto.rl.harness.codex.tb2_provider:modal_provider``."""
    del miles_args
    require_modal_client()
    backend = ModalSandboxBackend(
        app_name=os.environ.get(MODAL_APP_ENV, DEFAULT_MODAL_APP),
        ttl_s=int(_env_float(SANDBOX_TTL_ENV, DEFAULT_SANDBOX_TTL_S) or DEFAULT_SANDBOX_TTL_S),
        idle_timeout_s=int(_env_float(IDLE_TIMEOUT_ENV, DEFAULT_IDLE_TIMEOUT_S) or DEFAULT_IDLE_TIMEOUT_S),
        network_policy=NetworkPolicy.from_env(),
    )
    return _provider(backend)


def local_provider(miles_args: Any = None) -> Tb2EnvironmentProvider:
    """CPU-only provider (commands run on this host): smoke tests and dev."""
    del miles_args
    return _provider(LocalProcessBackend(network_policy=NetworkPolicy.from_env()))
