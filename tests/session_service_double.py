"""Test double of the session-service protocol (decoupling 3.8).

Implements the five paths of ``yeto/rl/harness/session_protocol.md`` on a
local aiohttp server (127.0.0.1, random port): create, chat (non-streaming,
``logprobs`` forced, ``meta_info.output_token_logprobs`` per choice), collect
samples, inspect, delete.  Records every request so tests can assert on it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

_CREATE_FIELDS = {"evaluation": bool, "temperature": float, "top_p": float, "top_k": int}


@dataclass
class _Session:
    params: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)


class SessionServiceDouble:
    def __init__(self, *, reply_tokens: tuple[int, ...] = (11, 12, 13)) -> None:
        self.sessions: dict[str, _Session] = {}
        self.requests: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.reply_tokens = reply_tokens
        self._ids = itertools.count(1)
        self._runner: web.AppRunner | None = None
        self.router = ""

    # -- lifecycle -----------------------------------------------------------
    async def __aenter__(self) -> "SessionServiceDouble":
        app = web.Application()
        app.router.add_post("/sessions", self._create)
        app.router.add_post("/sessions/{sid}/v1/chat/completions", self._chat)
        app.router.add_post("/sessions/{sid}/samples", self._samples)
        app.router.add_get("/sessions/{sid}", self._get)
        app.router.add_delete("/sessions/{sid}", self._delete)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.router = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    def session_url(self, session_id: str) -> str:
        return f"{self.router}/sessions/{session_id}"

    # -- handlers --------------------------------------------------------------
    def _session(self, request: web.Request) -> tuple[str, _Session]:
        sid = request.match_info["sid"]
        session = self.sessions.get(sid)
        if session is None:
            raise web.HTTPNotFound(text=f'{{"error": "session not found: session_id={sid}"}}',
                                   content_type="application/json")
        return sid, session

    async def _create(self, request: web.Request) -> web.Response:
        self.requests.append(("POST", "/sessions"))
        raw = await request.read()
        body = await request.json() if raw else {}
        if not isinstance(body, dict) or set(body) - set(_CREATE_FIELDS) or any(
                type(v) is not _CREATE_FIELDS[k] for k, v in body.items() if v is not None):
            return web.json_response({"error": "invalid CreateSessionRequest"}, status=400)
        sid = f"s{next(self._ids)}"
        self.sessions[sid] = _Session(params={"evaluation": False, **body})
        return web.json_response({"session_id": sid})

    async def _chat(self, request: web.Request) -> web.Response:
        sid, session = self._session(request)
        self.requests.append(("POST", f"/sessions/{sid}/v1/chat/completions"))
        body = dict(await request.json())
        body["logprobs"] = True  # forced, as the protocol requires
        tokens = list(self.reply_tokens)
        response = {
            "id": f"chatcmpl-{sid}-{len(session.records)}",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
                "meta_info": {"output_token_logprobs": [[-0.5, t, None] for t in tokens]},
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": len(tokens),
                      "total_tokens": 4 + len(tokens)},
        }
        session.records.append({"method": "POST", "path": "/v1/chat/completions",
                                "request": body, "response": response, "status_code": 200})
        return web.json_response(response)

    async def _samples(self, request: web.Request) -> web.Response:
        sid, session = self._session(request)
        self.requests.append(("POST", f"/sessions/{sid}/samples"))
        raw = await request.read()
        params = await request.json() if raw else {}
        tokens = [t for r in session.records
                  for t in (x[1] for x in r["response"]["choices"][0]["meta_info"]["output_token_logprobs"])]
        return web.json_response({"session_id": sid, "max_seq_len": params.get("max_seq_len"),
                                  "samples": [{"tokens": tokens, "records": len(session.records)}]})

    async def _get(self, request: web.Request) -> web.Response:
        sid, session = self._session(request)
        self.requests.append(("GET", f"/sessions/{sid}"))
        return web.json_response({"session_id": sid, "records": session.records, "metadata": {}})

    async def _delete(self, request: web.Request) -> web.Response:
        sid, _ = self._session(request)
        self.requests.append(("DELETE", f"/sessions/{sid}"))
        del self.sessions[sid]
        self.deleted.append(sid)
        return web.Response(status=204)
