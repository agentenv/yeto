"""Thin client for boat.dev sandboxes (S25 N10.8), plus the account-quota guard.

``BoatClient`` is the narrow, synchronous surface the boat provider needs.
:class:`SdkBoatClient` implements it on the official SDK (``boat-sdk``,
``boat_sdk.api.boat_api.BoatApi``); CPU tests pass a fake.  The SDK is
imported lazily, so the yeto core does not depend on it.

Credentials: ``BOAT_API_KEY``, else the token the boat CLI stores in
``~/.config/ascii/boat/config.json`` (``boat login``).  The token is never
logged or put in an exception message.

Quota (box_20 plan, read from ``GET /limits`` on 2026-10-19): 100 active
sandboxes, 12 starts per minute, 60 per hour, 200 per day.  :class:`BoatQuota`
holds the first two per process; :func:`check_in_flight_cap` refuses an
admission gate larger than the active-sandbox cap.  The hourly and daily caps
are left to the server (a refused create raises :class:`BoatQuotaError`).
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

API_HOST = "https://boat.dev/api/v1"
API_KEY_ENV = "BOAT_API_KEY"
CLI_CONFIG = Path("~/.config/ascii/boat/config.json")

# Plan limits (box_20).  Override per deployment with the env names below.
MAX_ACTIVE_SANDBOXES = 100
MAX_STARTS_PER_MINUTE = 12
MAX_ACTIVE_ENV = "YETO_BOAT_MAX_ACTIVE"
MAX_STARTS_ENV = "YETO_BOAT_STARTS_PER_MINUTE"

READY_STATES = frozenset({"ready", "idle", "running"})
FAILED_STATES = frozenset({"archiving", "archived", "error", "cancelled"})


class BoatError(RuntimeError):
    """A boat API call failed; Harbor reports it as the trial exception (ABORTED)."""


class BoatQuotaError(BoatError):
    """The account limit refused a create (HTTP 402/429) after the retries."""


@dataclass(frozen=True)
class CommandStatus:
    running: bool
    exit_code: int | None
    stdout: str
    stderr: str


class BoatClient(Protocol):
    """Synchronous boat operations used by :class:`~.boat_sandbox.BoatSandbox`."""

    def create(self, *, machine_type: str, ttl_seconds: int, idempotency_key: str) -> str: ...
    def state(self, sandbox_id: str) -> str: ...
    def start_command(self, sandbox_id: str, command: str) -> int: ...
    def command_status(self, sandbox_id: str, process_id: int) -> CommandStatus: ...
    def stop_command(self, sandbox_id: str, process_id: int) -> None: ...
    def put_file(self, sandbox_id: str, path: str, data: bytes) -> None: ...
    def get_file(self, sandbox_id: str, path: str) -> bytes: ...
    def usage_seconds(self, sandbox_id: str) -> float | None: ...
    def delete(self, sandbox_id: str) -> None: ...


def load_token() -> str:
    token = os.environ.get(API_KEY_ENV)
    if token:
        return token
    path = CLI_CONFIG.expanduser()
    try:
        token = json.loads(path.read_text()).get("token")
    except (OSError, ValueError) as exc:
        raise BoatError(f"no {API_KEY_ENV} and no readable boat CLI config at {path}") from exc
    if not token:
        raise BoatError(f"no {API_KEY_ENV} and no token in {path}; run `boat login`")
    return token


class SdkBoatClient:
    """:class:`BoatClient` on the official ``boat-sdk`` (one HTTP pool per instance)."""

    _CREATE_RETRY_STATUSES = frozenset({409, 429, 500, 502, 503, 504})

    def __init__(self, *, token: str | None = None, host: str = API_HOST, create_attempts: int = 5,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        from boat_sdk import ApiClient, Configuration
        from boat_sdk.api.boat_api import BoatApi

        self._client = ApiClient(Configuration(host=host, access_token=token or load_token()))
        self._api = BoatApi(self._client)
        self._create_attempts = create_attempts
        self._sleep = sleep

    def create(self, *, machine_type: str, ttl_seconds: int, idempotency_key: str) -> str:
        from boat_sdk.exceptions import ApiException
        from boat_sdk.models import CreateSandboxRequest

        # No account secrets, no snapshots (nothing to resume; delete erases it).
        request = CreateSandboxRequest(type=machine_type, ttl_seconds=ttl_seconds, no_env=True,
                                       snapshots=False)
        for attempt in range(self._create_attempts):
            try:
                # Same key on every attempt: a lost response never bills a second sandbox.
                return self._api.create(idempotency_key=idempotency_key,
                                        create_sandbox_request=request).sandbox.id
            except ApiException as exc:
                last = attempt == self._create_attempts - 1
                if exc.status not in self._CREATE_RETRY_STATUSES or last:
                    cls = BoatQuotaError if exc.status in (402, 429) else BoatError
                    raise cls(f"boat create failed: HTTP {exc.status} {exc.reason}") from None
                self._sleep(min(60.0, 5.0 * 2 ** attempt))
        raise AssertionError("unreachable")

    def state(self, sandbox_id: str) -> str:
        return str(self._api.get(sandbox_id).sandbox.state)

    def start_command(self, sandbox_id: str, command: str) -> int:
        from boat_sdk.models import CommandRequest

        started = self._api.command(sandbox_id, CommandRequest(command=command, detached=True)).actual_instance
        process_id = getattr(started, "process_id", None)
        if process_id is None:
            raise BoatError(f"boat did not start a detached command on {sandbox_id}")
        return int(process_id)

    def command_status(self, sandbox_id: str, process_id: int) -> CommandStatus:
        status = self._api.command_status(sandbox_id, process_id)
        return CommandStatus(running=bool(status.running), exit_code=status.exit_code,
                             stdout=status.stdout or "", stderr=status.stderr or "")

    def stop_command(self, sandbox_id: str, process_id: int) -> None:
        self._api.stop_command(sandbox_id, process_id)

    def put_file(self, sandbox_id: str, path: str, data: bytes) -> None:
        from boat_sdk.models import FileUrlRequest

        url = self._api.create_upload_url(sandbox_id, FileUrlRequest(path=path)).url
        _http(urllib.request.Request(url, data=data, method="PUT"))  # signed URL: no auth header

    def get_file(self, sandbox_id: str, path: str) -> bytes:
        from boat_sdk.models import FileUrlRequest

        url = self._api.create_download_url(sandbox_id, FileUrlRequest(path=path)).url
        return _http(urllib.request.Request(url, method="GET"))

    def usage_seconds(self, sandbox_id: str) -> float | None:
        # Billable seconds, type multiplier applied (small = 0.5).
        return float(self._api.usage(sandbox_id).seconds)

    def delete(self, sandbox_id: str) -> None:
        self._api.delete_sandbox(x_ascii_confirm_delete=sandbox_id, sandbox_id=sandbox_id)


def _http(request: urllib.request.Request, timeout_s: float = 600.0) -> bytes:
    # The signed URL is a secret: report the status only, never the URL.
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise BoatError(f"boat file transfer failed: HTTP {exc.code}") from None


def check_in_flight_cap(max_in_flight: int) -> int:
    """Refuse an admission gate that would hold more sandboxes than the boat account allows."""
    cap = int(os.environ.get(MAX_ACTIVE_ENV, MAX_ACTIVE_SANDBOXES))
    if max_in_flight > cap:
        raise ValueError(f"max_in_flight={max_in_flight} exceeds the boat active-sandbox cap {cap} "
                         f"({MAX_ACTIVE_ENV})")
    return max_in_flight


class BoatQuota:
    """Per-process guard: at most ``max_active`` live sandboxes, ``starts_per_minute`` creates.

    The active cap is a semaphore held from create to delete; the start rate
    is a sliding 60 s window.  Several processes share one account, so set the
    per-process caps so their sum stays within the plan.
    """

    def __init__(self, max_active: int, starts_per_minute: int, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if max_active <= 0 or starts_per_minute <= 0:
            raise ValueError("boat quota limits must be positive")
        self.max_active = max_active
        self.starts_per_minute = starts_per_minute
        self._active = asyncio.Semaphore(max_active)
        self._starts: collections.deque[float] = collections.deque()
        self._lock = asyncio.Lock()
        self._clock = clock

    @classmethod
    def from_env(cls) -> BoatQuota:
        return cls(int(os.environ.get(MAX_ACTIVE_ENV, MAX_ACTIVE_SANDBOXES)),
                   int(os.environ.get(MAX_STARTS_ENV, MAX_STARTS_PER_MINUTE)))

    async def acquire(self) -> None:
        """Take one active slot and one start; the caller must :meth:`release` the slot."""
        await self._active.acquire()
        try:
            async with self._lock:
                while True:
                    now = self._clock()
                    while self._starts and now - self._starts[0] >= 60.0:
                        self._starts.popleft()
                    if len(self._starts) < self.starts_per_minute:
                        self._starts.append(now)
                        return
                    await asyncio.sleep(60.0 - (now - self._starts[0]))
        except BaseException:
            self._active.release()
            raise

    def release(self) -> None:
        self._active.release()


_process_quota: BoatQuota | None = None
_process_client: SdkBoatClient | None = None


def process_client() -> SdkBoatClient:
    """One SDK client (one HTTP pool, thread-safe) for every boat sandbox of this process."""
    global _process_client
    if _process_client is None:
        _process_client = SdkBoatClient()
    return _process_client


def process_quota() -> BoatQuota:
    """The quota shared by every boat sandbox of this process (created on first use)."""
    global _process_quota
    if _process_quota is None:
        _process_quota = BoatQuota.from_env()
    return _process_quota
