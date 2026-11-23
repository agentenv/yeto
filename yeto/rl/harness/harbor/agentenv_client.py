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
# Cold-start root disk (sparse). Task images converted to overlaybd can report a
# base virtual size of 64 GiB; the gateway's default diskSizeMB (8 GiB) is then
# refused ("smaller than base virtual size; shrinking is disabled").
DISK_SIZE_MB_ENV = "YETO_AGENTENV_DISK_SIZE_MB"
DEFAULT_DISK_SIZE_MB = 131072
# Secure sandboxes (the default): AgentENV returns an ``envdAccessToken`` and envd
# (exec/files) refuses requests without it in ``X-Access-Token``.  Without
# secure mode anyone who can reach the gateway can run commands in the microVM.
# "1" opts out explicitly (e.g. a gateway too old to issue tokens).
INSECURE_ENV = "YETO_AGENTENV_INSECURE"
ENVD_TOKEN_HEADER = "X-Access-Token"

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


API_KEY_FILE_ENV = "AGENTENV_API_KEY_FILE"  # a 0600 file: AGENTENV_API_KEY=... (env-file) or the bare key


def key_from_text(text: str) -> str | None:
    """The key in an env-file (``AGENTENV_API_KEY=`` / ``E2B_API_KEY=``, ``export`` and quotes
    allowed) or a file holding only the key; None when there is none. Never logged."""
    bare: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        name, sep, value = line.partition("=")
        if sep:
            if name.strip() in (API_KEY_ENV, SDK_API_KEY_ENV):
                value = value.strip().strip("'\"")
                if value:
                    return value
            continue
        bare.append(line)
    return bare[0] if len(bare) == 1 and " " not in bare[0] else None


def load_token() -> str:
    token = os.environ.get(API_KEY_ENV) or os.environ.get(SDK_API_KEY_ENV)
    if not token and (path := os.environ.get(API_KEY_FILE_ENV, "").strip()):
        try:
            token = key_from_text(open(path).read())
        except OSError as error:
            raise AgentenvError(f"{API_KEY_FILE_ENV}={path}: {type(error).__name__}") from None
        if not token:
            raise AgentenvError(f"{API_KEY_FILE_ENV}={path} holds no {API_KEY_ENV}= line or bare key")
    if not token:
        raise AgentenvError(f"no {API_KEY_ENV} (or {SDK_API_KEY_ENV}, {API_KEY_FILE_ENV}) in the environment")
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

    Creates (template and cold) go through the control-plane REST with
    ``secure: true``: the E2B SDK (2.x) treats ``secure`` as deprecated and never
    sends it, and AgentENV creates an *insecure* sandbox when the field is
    absent.  The create response's ``envdAccessToken`` is kept per sandbox and
    every data-plane connection is checked to carry it as ``X-Access-Token``
    (fail closed).  The token is never logged or put in an exception.

    Verified against a real AgentENV gateway (envd behind a guard that rejects
    requests without ``X-Access-Token``): secure cold create from an OCI image,
    exec, file upload/download and delete.  Still unverified: ``kill`` stopping
    the whole guest process tree, usage, and the fork fan-out limit.
    """

    def __init__(self, *, api_key: str | None = None, api_url: str | None = None,
                 sandbox_url: str | None = None, request_timeout_s: float | None = None,
                 cold_start: bool | None = None, secure: bool | None = None) -> None:
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
        self._secure = (os.environ.get(INSECURE_ENV, "").strip() != "1"
                        if secure is None else secure)
        self._tokens: dict[str, str] = {}  # sandbox_id -> envdAccessToken (never logged)
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
            from e2b.exceptions import SandboxException
            try:
                box = Sandbox.connect(sandbox_id=sandbox_id, **self._opts())  # resumes a paused microVM
            except SandboxException as exc:
                raise self._wrap("connect", exc) from None
            self._check_secure(sandbox_id, box)
            self._boxes[sandbox_id] = box
        return box

    def _check_secure(self, sandbox_id: str, box) -> None:
        """Fail closed: in secure mode the envd connection must carry the access token.

        ``Sandbox.connect`` fetches the token from the authenticated control
        plane and sends it as ``X-Access-Token``; a sandbox that comes back
        without one was created insecure and is refused.
        """
        if not self._secure:
            return
        token = getattr(box, "_envd_access_token", None)
        if not isinstance(token, str) or not token:
            raise AgentenvError(f"agentenv sandbox {sandbox_id} has no envd access token "
                                f"(not a secure sandbox; set {INSECURE_ENV}=1 to allow)")
        self._tokens.setdefault(sandbox_id, token)

    def _adopt(self, box) -> str:
        self._check_secure(box.sandbox_id, box)
        self._boxes[box.sandbox_id] = box
        return box.sandbox_id

    # -- lifecycle ------------------------------------------------------------

    def create(self, *, template: str, ttl_seconds: int, metadata: dict[str, str] | None,
               env: dict[str, str] | None, idempotency_key: str) -> str:
        if self._cold:
            return self._create_cold(template, ttl_seconds, metadata, env)
        body: dict[str, object] = {"templateID": template, "timeout": ttl_seconds}
        if metadata:
            body["metadata"] = metadata
        if env:
            body["envVars"] = env
        return self._rest_create("/sandboxes", body, "create",
                                 headers={"Idempotency-Key": idempotency_key} if idempotency_key else None)

    def _create_cold(self, image: str, ttl_seconds: int, metadata: dict[str, str] | None,
                     env: dict[str, str] | None) -> str:
        """Build a microVM from a raw OCI image via the control plane ``/sandboxes-cold``.

        The SDK exposes no cold-start method; the body is AgentENV's
        ``NewColdSandbox`` (verified on a real gateway).  The returned id is
        handed to ``Sandbox.connect`` for the data plane on the first exec/file call.
        """
        body: dict[str, object] = {"image": image, "timeout": ttl_seconds,
                                   "cpuCount": int(os.environ.get("YETO_AGENTENV_CPU_COUNT", "2")),
                                   "memoryMB": int(os.environ.get("YETO_AGENTENV_MEMORY_MB", "4096")),
                                   "diskSizeMB": int(os.environ.get(DISK_SIZE_MB_ENV, DEFAULT_DISK_SIZE_MB))}
        if metadata:
            body["metadata"] = metadata
        if env:
            body["envVars"] = env
        return self._rest_create("/sandboxes-cold", body, "cold start")

    def _rest_create(self, path: str, body: dict[str, object], op: str, *,
                     headers: dict[str, str] | None = None) -> str:
        """POST a create to the control plane with ``secure`` set; returns the sandbox id.

        In secure mode the response must carry ``envdAccessToken`` (kept per
        sandbox, never logged); a gateway that ignores ``secure`` is refused.
        """
        import httpx

        base = self._api_url
        if not base:
            raise AgentenvError(f"agentenv {op} needs {API_URL_ENV} (the control-plane base URL)")
        body = {**body, "secure": self._secure}
        try:
            # trust_env=False: the control plane is a cluster endpoint, not a web
            # service to reach through the user's proxy.  Default httpx honours
            # ALL_PROXY/HTTPS_PROXY, so on a host with a Clash SOCKS proxy set this
            # call tries to go through SOCKS and crashes with an ImportError when
            # the 'socksio' extra is absent.
            resp = httpx.post(f"{base.rstrip('/')}{path}", json=body,
                              headers={**(headers or {}), "X-API-Key": self._api_key},
                              timeout=self._request_timeout_s, trust_env=False)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            cls = AgentenvQuotaError if status in (402, 429) else AgentenvError
            raise cls(f"agentenv {op} failed: HTTP {status}") from None
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise AgentenvError(f"agentenv {op} failed: {type(exc).__name__}") from None
        sandbox_id = data.get("sandboxID") or data.get("sandbox_id") if isinstance(data, dict) else None
        if not sandbox_id:
            raise AgentenvError(f"agentenv {op} returned no sandbox id")
        sandbox_id = str(sandbox_id)
        token = data.get("envdAccessToken")
        if self._secure:
            if not isinstance(token, str) or not token:
                self._kill_quietly(sandbox_id)
                raise AgentenvError(f"agentenv {op} returned no envd access token for {sandbox_id} "
                                    f"(gateway ignored secure; set {INSECURE_ENV}=1 to allow)")
            self._tokens[sandbox_id] = token
        return sandbox_id

    def _kill_quietly(self, sandbox_id: str) -> None:
        try:
            from e2b import Sandbox
            Sandbox.kill(sandbox_id=sandbox_id, **self._opts())
        except Exception:  # noqa: BLE001  best effort; the create already failed
            pass

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
        self._tokens.pop(sandbox_id, None)
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
        # timeout=0: the e2b background stream defaults to 60s total; the sandbox
        # layer enforces timeouts/cancel itself (it kills the command), so the
        # stream must not cap long agent (2400s) / verifier (300s) runs.
        handle = self._box(sandbox_id).commands.run(
            command, background=True, cwd=cwd, envs=env or None, user=_sdk_user(user),
            timeout=0)
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
