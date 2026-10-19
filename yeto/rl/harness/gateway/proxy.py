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

import json
import secrets
from typing import Any

from yeto.rl.compaction import CompactionConfig
from yeto.rl.engine.sampler_engine import TokenSamplerEngine
from yeto.rl.harness.chat_templates import ChatTemplate, TemplateError

from .core import AdmissionClosed, Gateway, GatewayConfig, GatewayError, SamplingOverrideError, TrajectoryInvalid
from .records import RecordingBackend
from .translate import UnsupportedShape

Message = dict[str, Any]


def chat_tools(entry: str, body: dict[str, Any]) -> list[Message]:
    """The request's tool schemas in Chat form (Responses tools are flat)."""
    tools = body.get("tools") or []
    if entry != "responses":
        return list(tools)
    out = []
    for tool in tools:
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
        return await self.gateway.handle(entry, body, trajectory_id=trajectory_id)

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
    # Reuse the Codex bridge's event sequence (created, item added/done, completed).
    from yeto.rl.harness.codex.codex_harness_agent import _sse_response

    return _sse_response(request={"model": request.get("model"), "tools": request.get("tools") or [],
                                  "prompt_cache_key": request.get("prompt_cache_key")},
                         output=response["output"], usage=response["usage"],
                         response_number=number)


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

