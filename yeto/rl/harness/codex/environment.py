"""Terminal environment surface used by the Codex driver (ports path).

The stock driver (``codex_harness_agent._AppServerDriver``) only needs two
calls from its environment: ``execute`` and ``submit``.  Legacy bound those to
the SecRLEnv ``EpisodeClient``; the Terminal-Bench path binds them to a sandbox
HTTP endpoint instead.  Verification (``evaluate``) is *not* part of this
surface: it belongs to the trusted layer (design D7 / R-D9).

``HttpTerminalEnvironment`` is the untrusted-side client (no HMAC key, bearer
token only).  ``serve_environment`` exposes any in-process environment over the
same wire so the worker subprocess can be exercised on CPU with a fake.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import aiohttp
from aiohttp import web

from .client import EpisodeAPIError, EpisodeTransportError

MAX_WIRE_BYTES = 1 << 20


class TerminalEnvironment(Protocol):
    async def execute(
        self, episode_id: str, command: str, timeout_seconds: float, output_bytes: int
    ) -> dict[str, Any]: ...

    async def submit(self, episode_id: str, submission: dict[str, Any]) -> dict[str, Any]: ...


class TrustedVerifier(Protocol):
    """Runs the task verifier for one episode; trusted layer only."""

    async def evaluate(self, episode_id: str) -> dict[str, Any]: ...


@dataclass
class FakeTerminalEnvironment:
    """Scripted environment for CPU tests: ``outputs[command]`` or default echo."""

    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    passed: bool = False
    calls: list[tuple[str, str]] = field(default_factory=list)
    submissions: list[dict[str, Any]] = field(default_factory=list)
    evaluations: int = 0
    exec_delay_seconds: float = 0.0

    async def execute(
        self, episode_id: str, command: str, timeout_seconds: float, output_bytes: int
    ) -> dict[str, Any]:
        self.calls.append((episode_id, command))
        if self.exec_delay_seconds:
            await asyncio.sleep(self.exec_delay_seconds)
        return dict(
            self.outputs.get(
                command,
                {"exit_code": 0, "output": f"ran:{command}", "timed_out": False, "truncated": False},
            )
        )

    async def submit(self, episode_id: str, submission: dict[str, Any]) -> dict[str, Any]:
        self.submissions.append(dict(submission))
        return {"accepted": True}

    async def evaluate(self, episode_id: str) -> dict[str, Any]:
        self.evaluations += 1
        return {"passed": self.passed, "testsh_rc": 0 if self.passed else 1}


class HttpTerminalEnvironment:
    """Untrusted-side client: ``POST {base}/execute`` and ``POST {base}/submit``."""

    def __init__(self, base_url: str, token: str, *, total_timeout_seconds: float = 600.0):
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = aiohttp.ClientTimeout(total=total_timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "HttpTerminalEnvironment":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _post(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise EpisodeTransportError("environment client is not started")
        try:
            async with self._session.post(
                f"{self._base}/{route}",
                json=payload,
                headers={"Authorization": f"Bearer {self._token}"},
            ) as response:
                raw = await response.content.read(MAX_WIRE_BYTES + 1)
                if len(raw) > MAX_WIRE_BYTES:
                    raise EpisodeTransportError("environment response is oversized")
                if response.status != 200:
                    raise EpisodeAPIError(
                        response.status, "environment_error", raw.decode("utf-8", "replace")[:512]
                    )
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise EpisodeTransportError("environment transport failed") from exc
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise EpisodeTransportError("environment returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise EpisodeTransportError("environment returned a non-object")
        return body

    async def execute(
        self, episode_id: str, command: str, timeout_seconds: float, output_bytes: int
    ) -> dict[str, Any]:
        return await self._post(
            "execute",
            {
                "episode_id": episode_id,
                "command": command,
                "timeout_seconds": timeout_seconds,
                "output_bytes": output_bytes,
            },
        )

    async def submit(self, episode_id: str, submission: dict[str, Any]) -> dict[str, Any]:
        return await self._post("submit", {"episode_id": episode_id, "submission": submission})


async def serve_environment(
    env: TerminalEnvironment, token: str, *, host: str = "127.0.0.1"
) -> tuple[web.AppRunner, str]:
    """Expose ``env`` on the worker wire. ``evaluate`` is deliberately not routed."""

    def _authorized(request: web.Request) -> bool:
        return hmac.compare_digest(
            request.headers.get("Authorization", ""), f"Bearer {token}"
        )

    async def execute(request: web.Request) -> web.Response:
        if not _authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        body = await request.json()
        try:
            result = await env.execute(
                str(body["episode_id"]),
                str(body["command"]),
                timeout_seconds=float(body["timeout_seconds"]),
                output_bytes=int(body["output_bytes"]),
            )
        except EpisodeAPIError as exc:
            return web.json_response({"error": exc.message}, status=exc.status)
        return web.json_response(result)

    async def submit(request: web.Request) -> web.Response:
        if not _authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        body = await request.json()
        try:
            result = await env.submit(str(body["episode_id"]), dict(body["submission"]))
        except EpisodeAPIError as exc:
            return web.json_response({"error": exc.message}, status=exc.status)
        return web.json_response(result)

    app = web.Application(client_max_size=MAX_WIRE_BYTES)
    app.router.add_post("/execute", execute)
    app.router.add_post("/submit", submit)
    runner = web.AppRunner(app, shutdown_timeout=1.0)
    await runner.setup()
    site = web.TCPSite(runner, host, 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001 - ephemeral port
    return runner, f"http://{host}:{port}"
