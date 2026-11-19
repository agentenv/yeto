"""Thin client for AgentENV (E2B-compatible) microVM sandboxes, plus the
account-quota guard.

AgentENV is a self-hosted sandbox platform (firecracker microVMs) whose
control plane speaks the E2B REST API, so the official E2B Python SDK (``e2b``,
pinned ``>=2.12``; developed against 2.64.0) drives it unchanged once the
gateway endpoint and API key are set.  Unlike boat (a VM that then runs
Docker), an AgentENV microVM *is* the task environment: exec and file transfer
land directly in the guest, with no ``docker exec`` / ``docker cp`` layer.

``AgentenvClient`` is the narrow, synchronous surface the provider needs.
:class:`SdkAgentenvClient` implements it on the SDK (calls run in a worker
thread); CPU tests pass a fake.  The SDK is imported lazily so the yeto core
never depends on it.

Endpoints are converged onto the client constructor (``api_url`` /
``sandbox_url`` / ``api_key`` / ``request_timeout``) rather than process-global
``E2B_*`` env vars, because ``E2B_API_URL`` / ``E2B_SANDBOX_URL`` are deprecated
in the SDK and this keeps the future migration to one place.  Credentials come
from ``AGENTENV_API_KEY`` (or ``E2B_API_KEY``); the key is never logged.

Dependency note: ``e2b>=2.12`` needs ``python>=3.10`` (``>=2.56`` needs 3.11);
a ``python3.9`` env only gets ``e2b<=2.10.2``, below the miles floor.  Pin
``e2b>=2.12`` in the agentenv deploy env (yeto runs on 3.13).
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from yeto.rl.agentic.loop_local import LoopLocal

API_KEY_ENV = "AGENTENV_API_KEY"
SDK_API_KEY_ENV = "E2B_API_KEY"  # the SDK's own name; accepted as a fallback
API_URL_ENV = "AGENTENV_API_URL"
SDK_API_URL_ENV = "E2B_API_URL"
SANDBOX_URL_ENV = "AGENTENV_SANDBOX_URL"
SDK_SANDBOX_URL_ENV = "E2B_SANDBOX_URL"
# "1": build microVMs from raw OCI images via the control plane /sandboxes-cold
# REST (the SDK has no cold-start method); default is prebuilt templates.
COLD_START_ENV = "YETO_AGENTENV_COLD_START"
REQUEST_TIMEOUT_ENV = "YETO_AGENTENV_REQUEST_TIMEOUT_S"

# Account caps.  AgentENV has no published fixed cap (TODO(verify on hardware):
# confirm the cluster's real concurrency/rate limits and tune these defaults;
# miles ran 16 concurrent creates on one 64-microVM node).
MAX_ACTIVE_SANDBOXES = 64
MAX_STARTS_PER_MINUTE = 60
MAX_ACTIVE_ENV = "YETO_AGENTENV_MAX_ACTIVE"
MAX_STARTS_ENV = "YETO_AGENTENV_STARTS_PER_MINUTE"
MAX_STARTS_HOUR_ENV = "YETO_AGENTENV_STARTS_PER_HOUR"  # unset = not held here

DEFAULT_TTL_S = 3 * 3600
DEFAULT_REQUEST_TIMEOUT_S = 120.0  # cold creates take tens of seconds
# AgentENV REST allows a fork fan-out of 1-100; the SDK caps it at 20.  Take
# the stricter bound.  TODO(verify on hardware): the real per-call limit.
MAX_FORK_COUNT = 20


class AgentenvError(RuntimeError):
    """An AgentENV API call failed; Harbor reports it as the trial exception (ABORTED)."""


class AgentenvQuotaError(AgentenvError):
    """The account limit refused a create / fork (rate limited) after the retries."""


@dataclass(frozen=True)
class CommandStatus:
    running: bool
    exit_code: int | None
    stdout: str
    stderr: str


class AgentenvClient(Protocol):
    """Synchronous AgentENV operations used by :class:`~.agentenv_sandbox.AgentenvSandbox`.

    Commands run in the background (the guest keeps a handle) so a cancelled or
    timed-out exec is stopped remotely with ``handle.kill()`` -- cancelling the
    coroutine alone does not stop the guest process (E2B semantics).
    """

    def create(self, *, template: str, ttl_seconds: int, metadata: dict[str, str] | None,
               env: dict[str, str] | None, idempotency_key: str) -> str: ...
    def fork(self, sandbox_id: str, count: int, *, ttl_seconds: int) -> list[str]: ...
    def start_command(self, sandbox_id: str, command: str, *, cwd: str | None,
                      env: dict[str, str] | None, user: str | int | None) -> int: ...
    def command_status(self, sandbox_id: str, process_id: int) -> CommandStatus: ...
    def stop_command(self, sandbox_id: str, process_id: int) -> None: ...
    def put_file(self, sandbox_id: str, path: str, data: bytes, *, user: str | int | None) -> None: ...
    def get_file(self, sandbox_id: str, path: str, *, user: str | int | None) -> bytes: ...
    def usage_seconds(self, sandbox_id: str) -> float | None: ...
    def set_ttl(self, sandbox_id: str, ttl_seconds: int) -> None: ...
    def delete(self, sandbox_id: str) -> None: ...
    # Optional on fakes: metadata-based orphan cleanup (there is no create name=)
    # def list_sandboxes(self, metadata: dict[str, str]) -> list[str]


def load_token() -> str:
    token = os.environ.get(API_KEY_ENV) or os.environ.get(SDK_API_KEY_ENV)
    if not token:
        raise AgentenvError(f"no {API_KEY_ENV} (or {SDK_API_KEY_ENV}) in the environment")
    return token


def _env_url(primary: str, fallback: str) -> str | None:
    return (os.environ.get(primary) or os.environ.get(fallback) or "").strip() or None


def _sdk_user(user: str | int | None) -> str | None:
    """Map a resolved user onto the E2B ``user`` argument (``None`` = image default).

    E2B 2.64 takes ``user`` as an optional string; ``None`` lets envd pick the
    image's default user.  Treat ``root``/``0`` as ``"root"`` and any other
    explicit value as ``"user"``.  TODO(verify on hardware): a numeric uid or a
    non-root/user name cannot be passed through to E2B.
    """
    if user is None:
        return None
    if isinstance(user, int):
        return "root" if user == 0 else "user"
    return "root" if user == "root" else "user"


class SdkAgentenvClient:
    """:class:`AgentenvClient` on the ``e2b`` SDK, pointed at an AgentENV gateway.

    One instance keeps the connected :class:`e2b.Sandbox` objects (one envd
    data-plane connection each, for exec/files) keyed by id, plus the background
    command handles keyed by (id, pid).  Lifecycle calls (create/fork/kill/
    set_timeout/get_info) use the SDK's by-id static variants, so they need no
    cached connection.

    TODO(verify on hardware): the exact SDK behaviour against a real AgentENV
    gateway (data-plane routing via ``sandbox_url`` vs wildcard domain,
    background-command polling, ``kill`` actually stopping the guest process
    tree, cold-start REST shape, usage, fork fan-out limit) is unverified until
    the cluster is up.
    """

    def __init__(self, *, api_key: str | None = None, api_url: str | None = None,
                 sandbox_url: str | None = None, request_timeout_s: float | None = None,
                 cold_start: bool | None = None) -> None:
        from e2b import Sandbox  # noqa: F401  (import check; lazy so core has no e2b dep)

        self._api_key = api_key or load_token()
        self._api_url = api_url if api_url is not None else _env_url(API_URL_ENV, SDK_API_URL_ENV)
        self._sandbox_url = (sandbox_url if sandbox_url is not None
                             else _env_url(SANDBOX_URL_ENV, SDK_SANDBOX_URL_ENV))
        if request_timeout_s is None:
            request_timeout_s = float(os.environ.get(REQUEST_TIMEOUT_ENV, "") or DEFAULT_REQUEST_TIMEOUT_S)
        self._request_timeout_s = request_timeout_s
        self._cold = (os.environ.get(COLD_START_ENV, "").strip() == "1"
                      if cold_start is None else cold_start)
        self._boxes: dict[str, object] = {}  # sandbox_id -> e2b.Sandbox (data plane)
        self._handles: dict[tuple[str, int], object] = {}  # (sandbox_id, pid) -> CommandHandle
        self._done: dict[tuple[str, int], CommandStatus] = {}

    # -- connection opts ------------------------------------------------------

    def _opts(self) -> dict[str, object]:
        opts: dict[str, object] = {"api_key": self._api_key, "request_timeout": self._request_timeout_s}
        if self._api_url:
            opts["api_url"] = self._api_url
        if self._sandbox_url:
            opts["sandbox_url"] = self._sandbox_url
        return opts

    def _box(self, sandbox_id: str):
        box = self._boxes.get(sandbox_id)
        if box is None:
            from e2b import Sandbox
            box = Sandbox.connect(sandbox_id=sandbox_id, **self._opts())  # resumes a paused microVM
            self._boxes[sandbox_id] = box
        return box

    def _adopt(self, box) -> str:
        self._boxes[box.sandbox_id] = box
        return box.sandbox_id

    # -- lifecycle ------------------------------------------------------------

    def create(self, *, template: str, ttl_seconds: int, metadata: dict[str, str] | None,
               env: dict[str, str] | None, idempotency_key: str) -> str:
        if self._cold:
            return self._create_cold(template, ttl_seconds, metadata, env)
        from e2b import Sandbox
        from e2b.exceptions import SandboxException

        try:
            box = Sandbox.create(template=template, timeout=ttl_seconds, metadata=metadata or None,
                                 envs=env or None, idempotency_key=idempotency_key, **self._opts())
        except SandboxException as exc:
            raise self._wrap("create", exc) from None
        return self._adopt(box)

    def _create_cold(self, image: str, ttl_seconds: int, metadata: dict[str, str] | None,
                     env: dict[str, str] | None) -> str:
        """Build a microVM from a raw OCI image via the control plane ``/sandboxes-cold``.

        The SDK exposes no cold-start method, so this is the one place the client
        speaks the control-plane REST directly; the returned id is then handed to
        ``Sandbox.connect`` for the data plane on the first exec/file call.
        TODO(verify on hardware): the exact ``NewColdSandbox`` body, the base URL
        shape, and whether AgentENV instead accepts an OCI ref as a ``template``.
        """
        import httpx

        base = self._api_url
        if not base:
            raise AgentenvError("cold start needs AGENTENV_API_URL (the control-plane base URL)")
        body: dict[str, object] = {"image": image, "timeout": ttl_seconds}
        if metadata:
            body["metadata"] = metadata
        if env:
            body["envVars"] = env
        try:
            # trust_env=False: the control plane is a cluster endpoint, not a web
            # service to reach through the user's proxy.  Default httpx honours
            # ALL_PROXY/HTTPS_PROXY, so on a host with a Clash SOCKS proxy set this
            # call tries to go through SOCKS and crashes with an ImportError when
            # the 'socksio' extra is absent.
            resp = httpx.post(f"{base.rstrip('/')}/sandboxes-cold", json=body,
                              headers={"X-API-Key": self._api_key}, timeout=self._request_timeout_s,
                              trust_env=False)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            cls = AgentenvQuotaError if status in (402, 429) else AgentenvError
            raise cls(f"agentenv cold start failed: HTTP {status}") from None
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise AgentenvError(f"agentenv cold start failed: {type(exc).__name__}") from None
        sandbox_id = data.get("sandboxID") or data.get("sandbox_id")
        if not sandbox_id:
            raise AgentenvError("agentenv cold start returned no sandbox id")
        return str(sandbox_id)

    def fork(self, sandbox_id: str, count: int, *, ttl_seconds: int) -> list[str]:
        """Snapshot-fork ``count`` new microVMs off ``sandbox_id`` (AgentENV ``/fork``).

        The source's id/TTL are unchanged; each fork is a pristine clone.  The
        SDK returns a list whose items are a :class:`e2b.Sandbox` on success or
        an ``Exception`` on failure (e.g. a rate limit), so they are judged one
        by one.  A fork that is all failures raises.
        """
        if count < 1:
            return []
        count = min(count, MAX_FORK_COUNT)
        from e2b import Sandbox
        from e2b.exceptions import SandboxException

        try:
            seed = self._box(sandbox_id)
            results = seed.fork(count=count, timeout=ttl_seconds)
        except SandboxException as exc:
            raise self._wrap("fork", exc) from None
        ids, last_exc = [], None
        for item in results:
            if isinstance(item, BaseException):
                last_exc = item
                continue
            ids.append(self._adopt(item))
        if not ids:
            raise AgentenvQuotaError(f"agentenv fork produced no sandboxes "
                                     f"({type(last_exc).__name__ if last_exc else 'unknown'})")
        return ids

    def set_ttl(self, sandbox_id: str, ttl_seconds: int) -> None:
        from e2b import Sandbox
        Sandbox.set_timeout(sandbox_id=sandbox_id, timeout=int(ttl_seconds), **self._opts())

    def usage_seconds(self, sandbox_id: str) -> float | None:
        """Approximate billable seconds from ``SandboxInfo.started_at`` / ``end_at``.

        TODO(verify on hardware): AgentENV may expose a more precise figure via
        ``/sandboxes/{id}/metrics``; ``started_at..end_at`` is the SDK's best.
        """
        from e2b import Sandbox
        try:
            info = Sandbox.get_info(sandbox_id=sandbox_id, **self._opts())
            started, end = getattr(info, "started_at", None), getattr(info, "end_at", None)
            if started is None:
                return None
            finish = end if end is not None else _now_like(started)
            return max(0.0, (finish - started).total_seconds())
        except Exception:  # noqa: BLE001  usage is a metric; never fail the caller
            return None

    def delete(self, sandbox_id: str) -> None:
        self._boxes.pop(sandbox_id, None)
        for key in [k for k in self._handles if k[0] == sandbox_id]:
            self._handles.pop(key, None)
            self._done.pop(key, None)
        from e2b import Sandbox
        from e2b.exceptions import SandboxException

        try:
            Sandbox.kill(sandbox_id=sandbox_id, **self._opts())
        except SandboxException as exc:
            raise self._wrap("delete", exc) from None

    # -- exec (background + poll, so cancel/timeout can kill the command) -----

    def start_command(self, sandbox_id: str, command: str, *, cwd: str | None,
                      env: dict[str, str] | None, user: str | int | None) -> int:
        handle = self._box(sandbox_id).commands.run(
            command, background=True, cwd=cwd, envs=env or None, user=_sdk_user(user))
        pid = int(handle.pid)
        self._handles[(sandbox_id, pid)] = handle
        return pid

    def command_status(self, sandbox_id: str, process_id: int) -> CommandStatus:
        key = (sandbox_id, process_id)
        if key in self._done:
            return self._done[key]
        handle = self._handles.get(key)
        if handle is None:
            raise AgentenvError(f"agentenv has no command {process_id} on {sandbox_id}")
        from e2b import CommandExitException

        # ``wait`` blocks until the background command exits; the sandbox layer
        # runs this in a worker thread and kills the command on cancel/timeout.
        try:
            result = handle.wait()
            status = CommandStatus(running=False, exit_code=result.exit_code,
                                   stdout=result.stdout, stderr=result.stderr)
        except CommandExitException as exc:
            status = CommandStatus(running=False, exit_code=exc.exit_code,
                                   stdout=exc.stdout, stderr=exc.stderr)
        self._done[key] = status
        self._handles.pop(key, None)
        return status

    def stop_command(self, sandbox_id: str, process_id: int) -> None:
        handle = self._handles.pop((sandbox_id, process_id), None)
        if handle is not None:
            handle.kill()  # cancelling the coroutine does not stop the guest process
        else:
            # No live handle (e.g. after a restart): kill by pid directly.
            try:
                self._box(sandbox_id).commands.kill(process_id)
            except Exception:  # noqa: BLE001
                pass

    # -- files ----------------------------------------------------------------

    def put_file(self, sandbox_id: str, path: str, data: bytes, *, user: str | int | None) -> None:
        self._box(sandbox_id).files.write(path, data, user=_sdk_user(user))

    def get_file(self, sandbox_id: str, path: str, *, user: str | int | None) -> bytes:
        data = self._box(sandbox_id).files.read(path, format="bytes", user=_sdk_user(user))
        return bytes(data)  # read returns a bytearray for format="bytes"

    def _wrap(self, op: str, exc: Exception) -> AgentenvError:
        name = type(exc).__name__
        cls = AgentenvQuotaError if "RateLimit" in name else AgentenvError
        return cls(f"agentenv {op} failed: {name}")


def _now_like(reference):
    """``datetime.now`` in ``reference``'s timezone (started_at may be tz-aware)."""
    import datetime as _dt

    return _dt.datetime.now(tz=getattr(reference, "tzinfo", None))


def check_in_flight_cap(max_in_flight: int) -> int:
    """Refuse an admission gate that would hold more sandboxes than AgentENV allows."""
    cap = int(os.environ.get(MAX_ACTIVE_ENV, MAX_ACTIVE_SANDBOXES))
    if max_in_flight > cap:
        raise ValueError(f"max_in_flight={max_in_flight} exceeds the agentenv active-sandbox cap "
                         f"{cap} ({MAX_ACTIVE_ENV})")
    return max_in_flight


class AgentenvQuota:
    """Per-process guard: at most ``max_active`` live sandboxes, ``starts_per_minute``
    creates (and ``starts_per_hour`` when given).

    The active cap is a semaphore held from create to delete; the start rates
    are sliding 60 s / 3600 s windows.  Several processes may share one account,
    so set the per-process caps so their sum stays within the cluster's budget.
    """

    def __init__(self, max_active: int, starts_per_minute: int, *, starts_per_hour: int | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if max_active <= 0 or starts_per_minute <= 0 or (starts_per_hour is not None and starts_per_hour <= 0):
            raise ValueError("agentenv quota limits must be positive")
        self.max_active = max_active
        self.starts_per_minute = starts_per_minute
        self.starts_per_hour = starts_per_hour
        self._active = asyncio.Semaphore(max_active)
        self._starts: collections.deque[float] = collections.deque()
        self._lock = asyncio.Lock()
        self._clock = clock

    @classmethod
    def from_env(cls) -> AgentenvQuota:
        hourly = os.environ.get(MAX_STARTS_HOUR_ENV, "").strip()
        return cls(int(os.environ.get(MAX_ACTIVE_ENV, MAX_ACTIVE_SANDBOXES)),
                   int(os.environ.get(MAX_STARTS_ENV, MAX_STARTS_PER_MINUTE)),
                   starts_per_hour=int(hourly) if hourly else None)

    def _wait_s(self, now: float) -> float:
        horizon = 3600.0 if self.starts_per_hour is not None else 60.0
        while self._starts and now - self._starts[0] >= horizon:
            self._starts.popleft()
        wait = 0.0
        last_minute = [t for t in self._starts if now - t < 60.0]
        if len(last_minute) >= self.starts_per_minute:
            wait = max(wait, 60.0 - (now - last_minute[-self.starts_per_minute]))
        if self.starts_per_hour is not None and len(self._starts) >= self.starts_per_hour:
            wait = max(wait, 3600.0 - (now - self._starts[-self.starts_per_hour]))
        return wait

    async def acquire(self) -> None:
        await self._active.acquire()
        try:
            async with self._lock:
                while True:
                    now = self._clock()
                    wait = self._wait_s(now)
                    if wait <= 0.0:
                        self._starts.append(now)
                        return
                    await asyncio.sleep(wait)
        except BaseException:
            self._active.release()
            raise

    def release(self) -> None:
        self._active.release()


_process_client: SdkAgentenvClient | None = None


def process_client() -> SdkAgentenvClient:
    """One SDK client (one connection pool) for every AgentENV sandbox of this process."""
    global _process_client
    if _process_client is None:
        _process_client = SdkAgentenvClient()
    return _process_client


_process_quota: LoopLocal[AgentenvQuota] = LoopLocal(AgentenvQuota.from_env)


def process_quota() -> AgentenvQuota:
    """The quota shared by every AgentENV sandbox of this process's event loop.

    It holds asyncio primitives, so it is kept per running loop
    (:class:`~yeto.rl.agentic.loop_local.LoopLocal`).  Call it inside a running loop.
    """
    return _process_quota.get()
