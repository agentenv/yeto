"""A fake SGLang HTTP server for CPU tests of weight updates during a generation.

It serves the native endpoints the ``sglang-server`` adapter calls and models
the semantics that matter for an update (read from the SGLang 0.5.20 source):

* a scheduler thread adds one token per tick to every running request; the
  token id is ``TOKEN_BASE + version`` so a test can tell which weights
  produced each token;
* ``/update_weights_from_disk`` on a running server waits until no request is
  running (the model-update writer lock), then loads; on a paused server it
  loads at once. With ``flush_cache`` true and queued requests it fails, as
  SGLang's cache flush asserts an idle scheduler;
* ``/pause_generation`` ``retract``: no tokens until ``/continue_generation``;
  ``abort``: every running request ends with finish_reason ``abort``;
* ``meta_info.weight_version`` is the version served when the request ended;
  with ``spans``, ``meta_info.weight_versions`` gives each output token's
  version as spans (the pinned SGLang fork).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

TOKEN_BASE = 100


@dataclass
class _Request:
    max_new_tokens: int
    tokens: list[int] = field(default_factory=list)
    versions: list[str] = field(default_factory=list)  # the version of each token
    finish: str | None = None
    weight_version: str = "default"
    retractions: int = 0
    done: threading.Event = field(default_factory=threading.Event)


def _spans(versions: list[str]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for i, version in enumerate(versions):
        if spans and spans[-1]["version"] == version:
            spans[-1]["end"] = i + 1
        else:
            spans.append({"version": version, "start": i, "end": i + 1})
    return spans


class FakeSGLang:
    def __init__(self, *, tick_s: float = 0.002, update_s: float = 0.05, spans: bool = True) -> None:
        self.tick_s = tick_s
        self.spans = spans  # meta_info.weight_versions (the pinned SGLang fork)
        self.update_s = update_s
        self.version = "default"
        self.paused = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.updates: list[dict[str, Any]] = []  # {"version", "waited_s"}
        self._lock = threading.Condition()
        self._running: list[_Request] = []
        self._stop = threading.Event()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self._server.server_address[1]
        self._threads = [threading.Thread(target=self._server.serve_forever, daemon=True),
                         threading.Thread(target=self._schedule, daemon=True)]

    def __enter__(self) -> "FakeSGLang":
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()

    def _version_int(self) -> int:
        return 0 if self.version == "default" else int(self.version)

    # ------------------------------------------------------------------ scheduler
    def _schedule(self) -> None:
        while not self._stop.wait(self.tick_s):
            with self._lock:
                if self.paused:
                    continue
                for request in list(self._running):
                    request.tokens.append(TOKEN_BASE + self._version_int())
                    request.versions.append(self.version)
                    if len(request.tokens) >= request.max_new_tokens:
                        self._end(request, "length")
                self._lock.notify_all()

    def _end(self, request: _Request, finish: str) -> None:
        request.finish = finish
        request.weight_version = self.version
        self._running.remove(request)
        request.done.set()

    # ------------------------------------------------------------------ endpoints
    def generate(self, body: dict[str, Any]) -> Any:
        texts = body.get("text", body.get("input_ids"))
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        params = body["sampling_params"]
        if int(params["max_new_tokens"]) == 0:  # teacher-forced score
            return [{"meta_info": {"input_token_logprobs": [[None, 1, None]] + [[-0.5, 2, None]] * (len(t) - 1)}}
                    for t in texts]
        requests = [_Request(int(params["max_new_tokens"])) for _ in texts]
        with self._lock:
            self._running.extend(requests)
        for request in requests:
            request.done.wait()
        out = [{"text": "x" * len(r.tokens),
                "meta_info": {"output_token_logprobs": [[-0.1, t, None] for t in r.tokens],
                              "finish_reason": {"type": r.finish}, "weight_version": r.weight_version,
                              "num_retractions": r.retractions,
                              **({"weight_versions": _spans(r.versions)} if self.spans else {})}}
               for r in requests]
        return out[0] if single else out

    def pause(self, body: dict[str, Any]) -> Any:
        with self._lock:
            self.paused = True
            if body.get("mode", "abort") == "abort":
                for request in list(self._running):
                    self._end(request, "abort")
            elif body["mode"] == "retract":
                for request in self._running:
                    request.retractions += 1
        return {"status": "ok"}

    def resume(self, body: dict[str, Any]) -> Any:
        with self._lock:
            self.paused = False
            self._lock.notify_all()
        return {"status": "ok"}

    def update(self, body: dict[str, Any]) -> tuple[int, Any]:
        began = time.monotonic()
        with self._lock:
            while not self.paused and self._running:  # the writer lock waits for running requests
                self._lock.wait(0.05)
            if body.get("flush_cache", True) and self._running:
                return 400, {"success": False, "message": "Cache flush failed after updating weights"}
            if body["model_path"].endswith("broken"):
                return 400, {"success": False, "message": "load failed"}
            time.sleep(self.update_s)
            self.version = str(body.get("weight_version") or self.version)
            self.updates.append({"version": self.version, "waited_s": time.monotonic() - began,
                                 "paused": self.paused})
        return 200, {"success": True, "message": "ok", "num_paused_requests": 0}

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _reply(self, status: int, data: Any) -> None:
                raw = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:  # noqa: N802
                self._reply(200 if self.path == "/health" else 404, {})

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                fake.calls.append((self.path, body))
                routes = {"/generate": lambda b: (200, fake.generate(b)),
                          "/pause_generation": lambda b: (200, fake.pause(b)),
                          "/continue_generation": lambda b: (200, fake.resume(b)),
                          "/update_weights_from_disk": fake.update}
                route = routes.get(self.path)
                self._reply(*(route(body) if route else (404, {})))

        return Handler
