"""Loopback-only review UI; writes require a session token and same-origin request."""
from __future__ import annotations

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .core import canonical
from .exporting import as_messages
from .store import Conflict, Store


def make_server(db_path, port=8765):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Avoid trace IDs and user content in web access logs.

        def send(self, status, value, mime="application/json", attachment=None):
            body = canonical(value).encode() if mime == "application/json" else value.encode()
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if not self.valid_host():
                return self.send(403, {"error": "Invalid Host"})
            parsed = urlparse(self.path)
            store = Store(db_path)
            try:
                if parsed.path == "/":
                    return self.send(200, (Path(__file__).parent / "static/review.html").read_text(), "text/html")
                if parsed.path == "/api/session":
                    return self.send(200, {"token": token, "generation_enabled": False})
                if parsed.path == "/api/gaps":
                    gaps = store.list_gaps()
                    if parse_qs(parsed.query).get("generated", ["0"])[0] == "1":
                        gaps = [g for g in gaps if g["status"] != "ungenerated"]
                    return self.send(200, {"gaps": gaps})
                if parsed.path.startswith("/api/gaps/"):
                    return self.send(200, store.detail(parsed.path.rsplit("/", 1)[1]))
                if parsed.path == "/api/export":
                    arm = parse_qs(parsed.query).get("arm", ["visible"])[0]
                    fmt = parse_qs(parsed.query).get("format", ["events"])[0]
                    if fmt not in {"events", "messages"}:
                        raise ValueError("Unknown export format")
                    rows = store.export(arm)
                    if fmt == "messages":
                        rows = (as_messages(row) for row in rows)
                    return self.send(200, "".join(canonical(row) + "\n" for row in rows), "application/x-ndjson", f"reviewed-{arm}-{fmt}.jsonl")
                return self.send(404, {"error": "Not found"})
            except (KeyError, ValueError) as exc:
                return self.send(409 if isinstance(exc, Conflict) else 400, {"error": str(exc)})
            finally:
                store.close()

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.valid_host() or self.headers.get("X-Cot-Token") != token or (origin and origin != "http://" + self.headers.get("Host", "")):
                return self.send(403, {"error": "Invalid session token or origin"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send(415, {"error": "Expected application/json"})
            store = Store(db_path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 131072:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                parts = urlparse(self.path).path.strip("/").split("/")
                if parts == ["api", "candidates", "import"]:
                    if not isinstance(body, dict) or set(body) != {"candidates"}:
                        raise ValueError("Expected a candidates array only")
                    imported = store.import_candidates(body["candidates"], {"source": "local_review_api"})
                    return self.send(200, {"imported": len(imported), "candidates": imported, "inference_started": False, "automatic_approval": False})
                if len(parts) != 4 or parts[:2] != ["api", "gaps"]:
                    return self.send(404, {"error": "Not found"})
                gap_id, action = parts[2:]
                if action == "review":
                    return self.send(200, store.review(gap_id, body["revision"], body["status"], body.get("text"), body.get("review_note", ""), body.get("review_tags", [])))
                if action == "regenerate":
                    store.request_regeneration(gap_id, body["revision"])
                    return self.send(200, {"queued": True, "inference_started": False})
                return self.send(404, {"error": "Not found"})
            except (KeyError, TypeError, ValueError) as exc:
                return self.send(409 if isinstance(exc, Conflict) else 400, {"error": str(exc)})
            finally:
                store.close()

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(db_path, port):
    server = make_server(db_path, port)
    print(f"Review UI: http://127.0.0.1:{server.server_port} (loopback only; no automatic inference)", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
