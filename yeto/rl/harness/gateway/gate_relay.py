"""HTTP relay that holds an agent's model calls at a closed :class:`~yeto.rl.agentic.suspend.TurnGate` (S26 E9.5).

On Miles the Harbor agent talks to Miles' session server directly, so yeto
has no place to hold a model call between two rounds.  :class:`GateRelay`
is that place: the agent gets ``<relay>/r/<token>`` instead of the session
URL, and the relay forwards every request (any method and path below the
token) to the session URL unchanged.  Before forwarding it waits while the
gate is closed; a reply with a retry status (default 503: the engine aborted
the turn at the cut-off) is sent again after the gate opens, when the gate
was closed during the call.

A held streaming call (``"stream": true`` in the JSON body) gets its HTTP
headers and SSE comment lines (``: held``) every ``keepalive_s`` seconds while
it waits, so the agent's idle-stream timeout does not fire during a training
step.  When the upstream then answers with an error status, the relay ends
the stream without a body; the agent sees a broken stream and retries the
call (its own retry policy).

The relay is protocol-agnostic and records nothing; the session server
records the tokens.  It runs on the event loop of the agent functions (the
gate's loop).
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from yeto.rl.agentic.suspend import SuspendLedger, TurnGate

# Hop-by-hop or recomputed headers that are never copied in either direction.
_DROP_HEADERS = frozenset({"host", "content-length", "transfer-encoding", "connection", "keep-alive",
                           "content-encoding", "upgrade", "proxy-connection"})
MIN_SECRET_CHARS = 24


class RelayError(RuntimeError):
    pass


@dataclass
class _Route:
    upstream: str
    ledger: SuspendLedger | None
    calls: int = field(default=0)


def _is_stream(body: bytes) -> bool:
    try:
        payload = json.loads(body) if body else None
    except (ValueError, UnicodeDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


class GateRelay:
    """Forwarding relay in front of the session server (module doc)."""

    def __init__(self, gate: TurnGate, *, host: str = "127.0.0.1", port: int | Sequence[int] = 0,
                 public_host: str | None = None, public_base: str | None = None,
                 path_secret: str | None = None, retry_statuses: tuple[int, ...] = (503,),
                 max_retries: int = 3, keepalive_s: float = 15.0, timeout_s: float = 3600.0) -> None:
        if path_secret is not None and len(path_secret) < MIN_SECRET_CHARS:
            raise RelayError(f"the path secret must have at least {MIN_SECRET_CHARS} characters")
        if max_retries < 0 or keepalive_s <= 0:
            raise RelayError("max_retries must be >= 0 and keepalive_s > 0")
        self.gate = gate
        # Several rollout processes on one machine share a port range: each binds
        # the first free port of ``port`` (a sequence); 0 = any free port.
        self.host = host
        self.ports = (int(port),) if isinstance(port, int) else tuple(int(p) for p in port)
        if not self.ports:
            raise RelayError("no relay port given")
        if public_base and len(self.ports) > 1:
            raise RelayError("a fixed public base URL needs one fixed relay port, not a range")
        self.port: int | None = None  # the bound port
        self.public_host = public_host or host
        self.public_base = public_base.rstrip("/") if public_base else None
        self._prefix = f"/k/{path_secret}" if path_secret else ""
        self.retry_statuses = frozenset(retry_statuses)
        self.max_retries = max_retries
        self.keepalive_s = keepalive_s
        self.timeout_s = timeout_s
        self._routes: dict[str, _Route] = {}
        self._runner: Any = None
        self._client: Any = None
        self._base_url: str | None = None
        self._start_lock = asyncio.Lock()

    # ------------------------------------------------------------ lifecycle
    async def _base(self) -> str:
        async with self._start_lock:
            if self._base_url is None:
                import aiohttp
                from aiohttp import web

                app = web.Application(client_max_size=64 * 1024**2)
                app.router.add_route("*", self._prefix + "/r/{token}/{tail:.*}", self._handle)
                runner = web.AppRunner(app, access_log=None)
                await runner.setup()
                await self._bind(web, runner)
                self._runner = runner
                self._client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s))
                self.port = port = runner.addresses[0][1]
                self._base_url = (self.public_base or f"http://{self.public_host}:{port}") + self._prefix
            return self._base_url

    async def _bind(self, web: Any, runner: Any) -> None:
        last: OSError | None = None
        for port in self.ports:
            try:
                await web.TCPSite(runner, self.host, port).start()
                return
            except OSError as exc:  # in use by another rollout process
                last = exc
        await runner.cleanup()
        raise RelayError(f"no free relay port in {self.ports[0]}..{self.ports[-1]} on {self.host}: {last}")

    async def open(self, upstream: str, ledger: SuspendLedger | None = None) -> str:
        """The URL the agent uses instead of ``upstream`` (the session server's OpenAI base)."""
        base = await self._base()
        token = secrets.token_urlsafe(16)
        self._routes[token] = _Route(upstream.rstrip("/"), ledger)
        return f"{base}/r/{token}"

    def close(self, url: str) -> None:
        self._routes.pop(url.rstrip("/").rsplit("/", 1)[-1], None)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._runner is not None:
            await self._runner.cleanup()
        self._runner = self._client = self._base_url = None

    # ------------------------------------------------------------ forwarding
    async def _handle(self, request: Any) -> Any:
        from aiohttp import web

        route = self._routes.get(request.match_info["token"])
        if route is None:
            return web.json_response({"error": {"message": "unknown relay route"}}, status=404)
        route.calls += 1
        body = await request.read()
        tail = request.match_info["tail"]
        url = f"{route.upstream}/{tail}" if tail else route.upstream
        if request.query_string:
            url = f"{url}?{request.query_string}"
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP_HEADERS}
        held: Any = None  # the open stream response once keep-alive started
        retries = 0
        while True:
            if self.gate.closed:
                if held is None and _is_stream(body):
                    held = await self._start_keepalive(request)
                await self._wait(held, route.ledger)
            epoch = self.gate.epoch
            try:
                status, reply_headers, payload = await self._forward(request.method, url, headers, body)
            except Exception:  # noqa: BLE001 - transport failure: retried only across a cut-off
                if self._suspended_since(epoch) and retries < self.max_retries:
                    retries += 1
                    self._count_retry(route.ledger)
                    continue
                if held is not None:
                    return await self._end_held(held)
                raise
            if status in self.retry_statuses and self._suspended_since(epoch) and retries < self.max_retries:
                retries += 1
                self._count_retry(route.ledger)
                continue
            if held is not None:
                if status >= 400:
                    return await self._end_held(held)
                await held.write(payload)
                await held.write_eof()
                return held
            return web.Response(status=status, body=payload, headers=reply_headers)

    def _suspended_since(self, epoch: int) -> bool:
        return self.gate.closed or self.gate.epoch != epoch

    @staticmethod
    def _count_retry(ledger: SuspendLedger | None) -> None:
        if ledger is not None:
            ledger.retried_turns += 1

    async def _forward(self, method: str, url: str, headers: dict[str, str],
                       body: bytes) -> tuple[int, dict[str, str], bytes]:
        async with self._client.request(method, url, headers=headers, data=body) as resp:
            payload = await resp.read()
            kept = {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_HEADERS}
            return resp.status, kept, payload

    async def _start_keepalive(self, request: Any) -> Any:
        from aiohttp import web

        held = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                       "Cache-Control": "no-cache"})
        await held.prepare(request)
        return held

    async def _wait(self, held: Any, ledger: SuspendLedger | None) -> None:
        if held is None:
            await self.gate.wait_open(ledger)
            return
        waiter = asyncio.ensure_future(self.gate.wait_open(ledger))
        try:
            while True:
                done, _ = await asyncio.wait({waiter}, timeout=self.keepalive_s)
                if done:
                    waiter.result()
                    return
                await held.write(b": held\n\n")
        finally:
            if not waiter.done():
                waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)

    @staticmethod
    async def _end_held(held: Any) -> Any:
        # Headers already said 200: the only error signal left is a stream that
        # ends without a reply; the agent retries the call.
        held.force_close()
        await held.write_eof()
        return held
