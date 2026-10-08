"""Loopback-only, GET-only dashboard server (design D2).

``yeto dashboard serve`` binds 127.0.0.1 (any non-loopback host is refused:
use an SSH tunnel), follows the tapes in a background thread and answers:

  GET /                         v7 page (same static file as the export)
  GET /api/overview
  GET /api/islands/<id>
  GET /api/rounds?only_bad=1
  GET /api/fleet
  GET /api/events?island=&type=&after=&limit=

Every other method gets 405. There is no endpoint that changes a run.
"""

from __future__ import annotations

import ipaddress
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from . import export as export_mod
from .reducer import Reducer
from .sources import TapeSource, discover, operator_stops_near

LOOPBACK_NAMES = ("localhost",)


def check_loopback(host: str) -> None:
    if host in LOOPBACK_NAMES:
        return
    try:
        ok = ipaddress.ip_address(host).is_loopback
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(
            f"refusing to listen on {host!r}: the dashboard only binds a loopback address "
            "(127.0.0.1); reach it from your laptop through an SSH tunnel, e.g. "
            "ssh -L 8787:127.0.0.1:8787 <head>")


class DashboardState:
    """Reducer + followed sources behind one lock."""

    def __init__(self, reducer: Reducer, paths: list[str]):
        self.reducer = reducer
        self.paths = list(paths)
        self.sources: dict[str, TapeSource] = {}
        self.lock = threading.Lock()
        self.last_poll: float | None = None

    def poll(self) -> int:
        n = 0
        for p in discover(self.paths):  # new island tapes appear mid-run
            key = str(p)
            if key not in self.sources:
                self.sources[key] = TapeSource(p)
        with self.lock:
            for src in self.sources.values():
                n += src.pump(self.reducer)
            for rec in operator_stops_near(self.paths):  # appears mid-run when we stop the app
                self.reducer.feed_operator_stop(rec)
            self.last_poll = time.time()
        return n


class _Handler(BaseHTTPRequestHandler):
    state: DashboardState
    live = True
    server_version = "yeto-dashboard"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _not_allowed(self) -> None:
        self._send(405, b'{"error":"read-only dashboard: only GET is allowed"}',
                   "application/json", {"Allow": "GET"})

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _not_allowed  # type: ignore

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        st = self.state
        r = st.reducer
        path = url.path.rstrip("/") or "/"
        with st.lock:
            if path in ("/", "/index.html"):
                body = export_mod.render_page(None).encode("utf-8")
                return self._send(200, body, "text/html; charset=utf-8")
            if path == "/api/view":  # everything the page draws (section 9)
                return self._json(r.page_view(live=self.live))
            if path == "/api/overview":
                return self._json(r.overview(live=self.live))
            if path.startswith("/api/islands/"):
                iid = unquote(path[len("/api/islands/"):])
                view = r.island_view(iid, live=self.live)
                if view is None:
                    return self._json({"error": f"unknown island {iid!r}"}, 404)
                return self._json(view)
            if path == "/api/rounds":
                rows = r.rounds()
                if q.get("only_bad") in ("1", "true", "yes"):
                    rows = [x for x in rows if x["bad"]]
                return self._json({"rounds": rows, "derived": True})
            if path == "/api/fleet":
                return self._json(r.fleet_view(live=self.live))
            if path == "/api/events":
                try:
                    after = int(q.get("after", 0))
                    limit = max(1, min(1000, int(q.get("limit", 200))))
                except ValueError:
                    return self._json({"error": "after/limit must be integers"}, 400)
                return self._json(r.events_view(island=q.get("island"), type=q.get("type"),
                                                after=after, limit=limit))
        self._json({"error": "not found"}, 404)


def make_server(state: DashboardState, host: str = "127.0.0.1", port: int = 8787,
                live: bool = True) -> ThreadingHTTPServer:
    check_loopback(host)
    handler = type("Handler", (_Handler,), {"state": state, "live": live})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def follow(state: DashboardState, stop: threading.Event, interval: float = 2.0) -> threading.Thread:
    def loop() -> None:
        while not stop.is_set():
            try:
                state.poll()
            except Exception as exc:  # keep serving what we have
                print(f"[dashboard] poll failed: {exc}")
            stop.wait(interval)

    t = threading.Thread(target=loop, name="dashboard-follow", daemon=True)
    t.start()
    return t


def tunnel_hint(port: int, head: str | None = None) -> str:
    return f"ssh -N -L {port}:127.0.0.1:{port} {head or '<head-host>'}"


def serve(paths: list[str], *, reducer: Reducer, host: str = "127.0.0.1", port: int = 8787,
          interval: float = 2.0, head: str | None = None) -> int:
    check_loopback(host)
    state = DashboardState(reducer, paths)
    n = state.poll()
    srv = make_server(state, host, port)
    stop = threading.Event()
    follow(state, stop, interval)
    print(f"[dashboard] {len(state.sources)} tape(s), {n} record(s); serving "
          f"http://{host}:{srv.server_address[1]}/ (read-only, GET only)", flush=True)
    for s in state.sources:
        print(f"[dashboard]   source {s}", flush=True)
    print(f"[dashboard] from your laptop: {tunnel_hint(srv.server_address[1], head)}"
          f"  then open http://127.0.0.1:{srv.server_address[1]}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        srv.server_close()
    return 0
