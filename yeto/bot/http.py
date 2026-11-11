"""Tiny JSON-over-HTTP helper (urllib) and the read-only dashboard client.

Everything that talks to the network takes an ``http`` callable with the
signature of :func:`request_json`, so tests swap in a fake.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable

HttpFn = Callable[..., Any]


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:200]}")
        self.status = status
        self.body = body


def multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """``files``: name -> (filename, bytes, content type). Returns (body, content type)."""
    boundary = "yeto-" + uuid.uuid4().hex
    out = []
    for k, v in (fields or {}).items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, (fname, data, ctype) in files.items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'
                   f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def request_json(method: str, url: str, *, body: Any = None, headers: dict | None = None,
                 timeout: float = 30.0, files: dict | None = None) -> Any:
    """JSON request (or multipart form when ``files`` is given, ``body`` then holds the form fields)."""
    data = None
    hdrs = dict(headers or {})
    if files:
        data, ctype = multipart(body or {}, files)
        hdrs["Content-Type"] = ctype
    elif body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json; charset=utf-8")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise HttpError(exc.code, exc.read().decode("utf-8", errors="replace"), redact_url(url)) from None
    return json.loads(raw) if raw else None


_URL_SECRETS = (
    (re.compile(r"/bot[^/?#]+"), "/bot<token>"),  # Telegram Bot API: /bot<token>/<method>
    (re.compile(r"/hook/[^/?#]+"), "/hook/<token>"),  # Lark custom bot: /open-apis/bot/v2/hook/<token>
)


def redact_url(url: str) -> str:
    """``url`` with the credentials some APIs carry in the path replaced by ``<token>``.
    Use it wherever a URL reaches a log line, an exception or the dry-run output."""
    for pat, repl in _URL_SECRETS:
        url = pat.sub(repl, url)
    return url


class DashboardClient:
    """GET-only client of ``yeto dashboard serve`` (never touches nodes or the run)."""

    def __init__(self, base_url: str, *, timeout: float = 10.0, http: HttpFn = request_json):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.http = http

    def _get(self, path: str) -> Any:
        return self.http("GET", self.base + path, timeout=self.timeout)

    def view(self) -> dict:
        return self._get("/api/view")

    def bad_rounds(self) -> list[dict]:
        return (self._get("/api/rounds?only_bad=1") or {}).get("rounds") or []
