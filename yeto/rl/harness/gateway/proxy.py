"""Records proxy: the gateway plus :class:`.records.RecordingBackend`, with an
HTTP shell (S25 N10.5).

Sits in front of a sampler engine. An agent that only speaks OpenAI (Harbor's
built-in codex agent uses Responses) gets a per-trajectory base URL::

    POST /trajectories                         -> {"trajectory_id", "base_url"}
    POST /t/{trajectory_id}/v1/responses       (Codex; stream:true answers SSE)
    POST /t/{trajectory_id}/v1/chat/completions
    GET  /t/{trajectory_id}/records            token ids, loss mask, log-probs, source
    POST /t/{trajectory_id}/close

The sampling fields stay signed by the gateway (a request may not change them).
Errors fail closed: an invalid trajectory answers 409, an unsupported request
400, a draining member 503.
"""

from __future__ import annotations

import copy
import json
import secrets
import time
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

    def open(self) -> str:
        return f"traj-{secrets.token_hex(6)}"

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

    # ------------------------------------------------------------------ HTTP
    def app(self, *, public_url: str = "") -> Any:
        from aiohttp import web

        async def create(request: web.Request) -> web.Response:
            trajectory_id = self.open()
            base = public_url or f"{request.scheme}://{request.host}"
            return web.json_response({"trajectory_id": trajectory_id, "base_url": f"{base}/t/{trajectory_id}/v1"})

        def entry_route(entry: str) -> Any:
            async def route(request: web.Request) -> web.StreamResponse:
                trajectory_id = request.match_info["tid"]
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
            return web.json_response(self.records(request.match_info["tid"]))

        async def close(request: web.Request) -> web.Response:
            self.close(request.match_info["tid"])
            return web.json_response({"closed": True})

        app = web.Application(client_max_size=64 * 1024**2)
        app.router.add_post("/trajectories", create)
        app.router.add_post("/t/{tid}/v1/responses", entry_route("responses"))
        app.router.add_post("/t/{tid}/v1/chat/completions", entry_route("chat"))
        app.router.add_get("/t/{tid}/records", records)
        app.router.add_post("/t/{tid}/close", close)
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

