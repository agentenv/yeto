"""Records proxy: the gateway plus :class:`.records.RecordingBackend`, with an
HTTP shell.

Sits in front of a sampler engine. An agent that only speaks OpenAI (Harbor's
built-in codex agent uses Responses) gets a per-trajectory base URL::

    POST /trajectories                         -> {"trajectory_id", "base_url"}   (control, opt-in)
    POST /t/{tid}/{token}/v1/responses         (Codex; stream:true answers SSE)
    POST /t/{tid}/{token}/v1/chat/completions
    GET  /t/{tid}/{token}/records              token ids, loss mask, log-probs, source (control, opt-in)
    POST /t/{tid}/{token}/close                (control, opt-in)

Access (threat model: every sandbox on a sampler is hostile to every other):

* each trajectory gets its own random ``token`` (:meth:`RecordsProxy.open`,
  256 bits) and every route under ``/t/{tid}/`` checks it in constant time;
  an unknown trajectory, a wrong token or another trajectory's token answer
  the same 403.  The trajectory id alone (often guessable) grants nothing,
  and a leaked token reaches one trajectory only, until it is forgotten.
* the control routes (create, records, close) are not served unless
  ``app(control=True)`` asks for them (never on a port a sandbox can reach);
  when served they also need a peer in
  ``control_peers`` (default loopback) that came without a forwarding header
  (a tunnel hop); :class:`~.recorder.ProxyTokenRecorder` serves no control
  route at all (it reads records in process), so a sandbox can only call the
  model routes of its own trajectory.
* the deployment's shared path secret (``/k/<secret>/``, recorder) stays as
  a second layer in front of all of this.

The sampling fields stay signed by the gateway (a request may not change them).
Errors fail closed: an invalid trajectory answers 409, an unsupported request
400, a draining member 503.
"""

from __future__ import annotations

import copy
import hmac
import ipaddress
import json
import secrets
import threading
import time
from collections.abc import Iterable
from contextvars import ContextVar
from typing import Any

from yeto.rl.compaction import CompactionConfig
from yeto.rl.engine.sampler_engine import TokenSamplerEngine
from yeto.rl.harness.chat_templates import ChatTemplate, TemplateError

from .core import AdmissionClosed, Gateway, GatewayConfig, GatewayError, SamplingOverrideError, TrajectoryInvalid
from .records import RecordingBackend
from .translate import OMITTED_TOOL_TYPES, UnsupportedShape

# The trajectory whose request is being served.  Engine calls run in a thread
# started by ``asyncio.to_thread``, which copies this context: an engine that
# owns no version of its own (verl's rollout) can tag what it saw per trajectory.
CURRENT_TRAJECTORY: ContextVar[str | None] = ContextVar("records_proxy_trajectory", default=None)

Message = dict[str, Any]

TOKEN_BYTES = 32
LOOPBACK_PEERS: tuple[str, ...] = ("127.0.0.0/8", "::1/128")
# A request that carries one of these came through a proxy or tunnel hop: its
# socket peer is the hop, not the caller, so it never passes the control check.
FORWARDING_HEADERS = ("Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Real-IP")


class TrajectoryForbidden(GatewayError):
    """Unknown trajectory, wrong token or a control call from outside ``control_peers`` (HTTP 403)."""


def chat_tools(entry: str, body: dict[str, Any]) -> list[Message]:
    """The request's tool schemas in Chat form (Responses tools are flat).
    Types in :data:`OMITTED_TOOL_TYPES` are left out of the prompt."""
    tools = body.get("tools") or []
    if entry != "responses":
        return list(tools)
    out = []
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") in OMITTED_TOOL_TYPES:
            continue
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise UnsupportedShape(f"unsupported tool {tool!r}")
        function = {k: tool[k] for k in ("name", "description", "parameters") if k in tool}
        out.append({"type": "function", "function": function})
    return out


class RecordsProxy:
    """Gateway + records backend for one engine and one chat template."""

    def __init__(self, engine: TokenSamplerEngine, template: ChatTemplate, *, model: str,
                 signed_sampling: dict[str, Any], sampler_id: str = "records-proxy",
                 compaction: CompactionConfig | None = None, max_chains: int | None = None,
                 harness_board: Any = None, member: str | None = None) -> None:
        self.backend = RecordingBackend(engine, template, sampler_id=sampler_id, compaction=compaction)
        self.gateway = Gateway(GatewayConfig(
            model=model, signed_sampling=dict(signed_sampling), max_chains=max_chains,
            keeps_history_reasoning=template.spec.keeps_history_reasoning,
            harness_board=harness_board, member=member), self.backend)
        self._responses = 0
        self._tokens: dict[str, str] = {}
        self._tokens_lock = threading.Lock()  # open/forget run on several event loops

    def open(self, trajectory_id: str | None = None) -> str:
        """Register a trajectory and give it a fresh random token (module doc); returns its id.

        An id that is already open is refused: a second opener would otherwise
        learn nothing but could still race the first one's state."""
        trajectory_id = trajectory_id or f"traj-{secrets.token_hex(8)}"
        if "/" in trajectory_id or not trajectory_id:
            raise TrajectoryForbidden(f"bad trajectory id {trajectory_id!r}")
        token = secrets.token_urlsafe(TOKEN_BYTES)
        with self._tokens_lock:
            if trajectory_id in self._tokens:
                raise TrajectoryForbidden(f"trajectory {trajectory_id!r} is already open")
            self._tokens[trajectory_id] = token
        return trajectory_id

    def token(self, trajectory_id: str) -> str:
        """The token of an open trajectory (the trainer side only; never logged)."""
        with self._tokens_lock:
            token = self._tokens.get(trajectory_id)
        if token is None:
            raise TrajectoryForbidden("trajectory is not open")
        return token

    def path(self, trajectory_id: str) -> str:
        """``/t/{tid}/{token}``: the prefix of the trajectory's routes."""
        return f"/t/{trajectory_id}/{self.token(trajectory_id)}"

    def authorized(self, trajectory_id: str, token: str) -> bool:
        """Constant-time check of ``token`` against the open trajectory's own token."""
        with self._tokens_lock:
            expected = self._tokens.get(trajectory_id)
        # Compare against a dummy for unknown ids so timing does not tell which ids are open.
        ok = hmac.compare_digest((expected or "\0" * 43).encode(), token.encode())
        return ok and expected is not None

    async def handle(self, entry: str, body: dict[str, Any], *, trajectory_id: str) -> dict[str, Any]:
        self.backend.declare_tools(trajectory_id, chat_tools(entry, body))
        token = CURRENT_TRAJECTORY.set(trajectory_id)
        try:
            return await self.gateway.handle(entry, body, trajectory_id=trajectory_id)
        finally:
            CURRENT_TRAJECTORY.reset(token)

    def records(self, trajectory_id: str) -> dict[str, Any]:
        out = self.backend.records(trajectory_id)
        out["gateway"] = self.gateway.snapshot(trajectory_id)
        return out

    def close(self, trajectory_id: str) -> None:
        self.gateway.close(trajectory_id)

    def forget(self, trajectory_id: str) -> None:
        """Drop the trajectory's state in the gateway and the backend (after close and records)."""
        self.gateway.forget(trajectory_id)
        self.backend.forget(trajectory_id)
        with self._tokens_lock:
            self._tokens.pop(trajectory_id, None)

    # ------------------------------------------------------------------ HTTP
    def app(self, *, public_url: str = "", control: bool = False,
            control_peers: Iterable[str] = LOOPBACK_PEERS) -> Any:
        """The HTTP shell (module doc): the model routes only, unless ``control=True``."""
        from aiohttp import web

        networks = tuple(ipaddress.ip_network(peer, strict=False) for peer in control_peers)

        def control_peer(request: web.Request) -> bool:
            if any(name in request.headers for name in FORWARDING_HEADERS):
                return False
            try:
                address = ipaddress.ip_address((request.remote or "").split("%", 1)[0])
            except ValueError:
                return False
            return any(address in net for net in networks)

        def check(request: web.Request, *, control_route: bool) -> str:
            trajectory_id = request.match_info["tid"]
            if control_route and not control_peer(request):
                raise TrajectoryForbidden("forbidden")
            if not self.authorized(trajectory_id, request.match_info["token"]):
                raise TrajectoryForbidden("forbidden")
            return trajectory_id

        async def create(request: web.Request) -> web.Response:
            if not control_peer(request):
                return _error(403, TrajectoryForbidden("forbidden"))
            trajectory_id = self.open()
            base = public_url or f"{request.scheme}://{request.host}"
            return web.json_response({"trajectory_id": trajectory_id,
                                      "base_url": f"{base}{self.path(trajectory_id)}/v1"})

        def entry_route(entry: str) -> Any:
            async def route(request: web.Request) -> web.StreamResponse:
                try:
                    trajectory_id = check(request, control_route=False)
                except TrajectoryForbidden as exc:
                    return _error(403, exc)
                try:
                    body = await request.json()
                    result = await self.handle(entry, body, trajectory_id=trajectory_id)
                except (UnsupportedShape, SamplingOverrideError, TemplateError, json.JSONDecodeError) as exc:
                    return _error(400, exc)
                except AdmissionClosed as exc:
                    return _error(503, exc)
                except (TrajectoryInvalid, GatewayError) as exc:
                    return _error(409, exc)
                if not body.get("stream"):
                    return web.json_response(result)
                self._responses += 1
                payload = _responses_sse(body, result, self._responses) if entry == "responses" else _chat_sse(result)
                return web.Response(body=payload, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            return route

        async def records(request: web.Request) -> web.Response:
            try:
                trajectory_id = check(request, control_route=True)
            except TrajectoryForbidden as exc:
                return _error(403, exc)
            return web.json_response(self.records(trajectory_id), dumps=_records_dumps)

        async def close(request: web.Request) -> web.Response:
            try:
                trajectory_id = check(request, control_route=True)
            except TrajectoryForbidden as exc:
                return _error(403, exc)
            self.close(trajectory_id)
            return web.json_response({"closed": True})

        app = web.Application(client_max_size=64 * 1024**2)
        app.router.add_post("/t/{tid}/{token}/v1/responses", entry_route("responses"))
        app.router.add_post("/t/{tid}/{token}/v1/chat/completions", entry_route("chat"))
        if control:
            app.router.add_post("/trajectories", create)
            app.router.add_get("/t/{tid}/{token}/records", records)
            app.router.add_post("/t/{tid}/{token}/close", close)
        return app


def _error(status: int, exc: Exception) -> Any:
    from aiohttp import web

    return web.json_response({"error": {"type": type(exc).__name__, "message": str(exc)}}, status=status)


def _responses_sse(request: dict[str, Any], response: dict[str, Any], number: int) -> bytes:
    """Responses API event stream: created, per-item added/(arguments)/done, completed.

    One event per output item, no token streaming.  ``instructions`` echoes the
    request (the client owns its system prompt).
    """
    output = response["output"]
    completed = {
        "id": f"resp_yeto_{number}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": request.get("instructions"),
        "max_output_tokens": None,
        "max_tool_calls": None,
        "model": request.get("model"),
        "output": output,
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "prompt_cache_key": request.get("prompt_cache_key"),
        "reasoning": {"effort": "xhigh", "summary": "none"},
        "safety_identifier": None,
        "service_tier": "default",
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": request.get("tools") or [],
        "top_logprobs": 0,
        "top_p": None,
        "truncation": "disabled",
        "usage": response["usage"],
        "user": None,
        "metadata": {},
    }
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {**completed, "status": "in_progress", "output": []}}]
    for index, item in enumerate(output):
        added = copy.deepcopy(item)
        added["status"] = "in_progress"
        if item["type"] == "function_call":
            added["arguments"] = ""
        events.append({"type": "response.output_item.added", "output_index": index, "item": added})
        if item["type"] == "function_call":
            events.append({"type": "response.function_call_arguments.delta", "item_id": item["id"],
                           "output_index": index, "delta": item["arguments"]})
            events.append({"type": "response.function_call_arguments.done", "item_id": item["id"],
                           "output_index": index, "arguments": item["arguments"]})
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    events.append({"type": "response.completed", "response": completed})
    chunks = [f"event: {event['type']}\n".encode() + b"data: " + _canonical_json(event) + b"\n\n"
              for event in events]
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _chat_sse(completion: dict[str, Any]) -> bytes:
    """One chunk with the whole message, then ``[DONE]`` (no token streaming)."""
    choice = completion["choices"][0]
    delta = {k: v for k, v in choice["message"].items() if k in ("role", "content", "reasoning_content", "tool_calls")}
    if delta.get("tool_calls"):
        delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(delta["tool_calls"])]
    chunk = {"id": completion.get("id", "chatcmpl"), "object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": delta, "finish_reason": choice["finish_reason"]}],
             "usage": completion.get("usage")}
    return b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n"


def _records_dumps(value: Any) -> str:
    """JSON of the records with the engine's rollout arrays as nested lists."""
    return json.dumps(value, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
