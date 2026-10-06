"""Authenticated async client for the loopback secrlenv episode daemon."""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Self
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

import aiohttp


class EpisodeClientError(RuntimeError):
    pass


class EpisodeAPIError(EpisodeClientError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"episode daemon returned {status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


class EpisodeTransportError(EpisodeClientError):
    pass


def _read_token() -> str:
    token_file = os.getenv("SECRLENV_BEARER_TOKEN_FILE")
    if token_file:
        path = Path(token_file)
        try:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                raise EpisodeClientError(
                    "SECRLENV_BEARER_TOKEN_FILE must be a private regular file"
                )
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise EpisodeClientError(f"cannot read episode daemon token file: {exc}") from exc
    else:
        token = os.getenv("SECRLENV_BEARER_TOKEN", "").strip()
    if len(token) < 32:
        raise EpisodeClientError("episode daemon bearer token is missing or too short")
    return token


def daemon_url_from_env() -> str:
    value = os.getenv("SECRLENV_DAEMON_URL", "http://127.0.0.1:8765").rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.port is None
    ):
        raise EpisodeClientError(
            "SECRLENV_DAEMON_URL must be an unauthenticated loopback HTTP origin"
        )
    return value


def require_daemon_ready(*, timeout_seconds: float = 10.0) -> None:
    """Fail before GPU initialization unless the expected daemon is healthy."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("episode daemon readiness timeout must be positive")
    try:
        with urlopen(
            f"{daemon_url_from_env()}/healthz", timeout=timeout_seconds
        ) as response:
            if response.status != 200:
                raise EpisodeTransportError(
                    f"secrlenv episode daemon health check returned {response.status}"
                )
            value = json.loads(response.read())
    except EpisodeTransportError:
        raise
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
        raise EpisodeTransportError(
            "secrlenv episode daemon is not healthy on its configured loopback origin"
        ) from exc
    if not isinstance(value, dict) or value.get("ok") is not True:
        raise EpisodeTransportError("secrlenv episode daemon health response is invalid")
    expected_task_pack = os.getenv("SECRLENV_TASK_PACK_SHA256", "").strip()
    if expected_task_pack and value.get("task_pack_sha256") != expected_task_pack:
        raise EpisodeTransportError(
            "secrlenv episode daemon task-pack identity does not match the run plan"
        )


class EpisodeClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        *,
        total_timeout_seconds: float = 1200.0,
    ) -> None:
        self.base_url = (base_url or daemon_url_from_env()).rstrip("/")
        # Apply the same validation to explicit constructor values used in tests
        # and direct-drive tools, not only to the environment-derived default.
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.port is None
        ):
            raise EpisodeClientError("episode daemon URL must be a loopback HTTP origin")
        self.token = token or _read_token()
        if len(self.token) < 32:
            raise EpisodeClientError("episode daemon bearer token is too short")
        self.timeout = aiohttp.ClientTimeout(total=total_timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> Self:
        self._session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=aiohttp.TCPConnector(limit=4),
            headers={"Authorization": f"Bearer {self.token}"},
        )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._session is None:
            raise EpisodeClientError("EpisodeClient must be used as an async context manager")
        try:
            async with self._session.request(
                method,
                f"{self.base_url}{path}",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as response:
                raw = await response.text()
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise EpisodeTransportError(f"episode daemon request failed: {exc}") from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EpisodeTransportError("episode daemon returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise EpisodeTransportError("episode daemon returned a non-object response")
        if not 200 <= response.status < 300:
            error = value.get("error") if isinstance(value.get("error"), dict) else {}
            raise EpisodeAPIError(
                response.status,
                str(error.get("code", "unknown_error")),
                str(error.get("message", "request failed")),
            )
        return value

    async def create(self, task_id: str, prompt_tier: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/v1/episodes",
            {"task_id": task_id, "prompt_tier": prompt_tier},
        )

    async def execute(
        self,
        episode_id: str,
        command: str,
        *,
        timeout_seconds: float,
        output_bytes: int,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/episodes/{episode_id}/exec",
            {
                "command": command,
                "timeout_seconds": timeout_seconds,
                "output_bytes": output_bytes,
            },
        )

    async def submit(self, episode_id: str, submission: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/episodes/{episode_id}/submit",
            {"submission": submission},
        )

    async def evaluate(self, episode_id: str) -> dict[str, Any]:
        return await self._request(
            "POST", f"/v1/episodes/{episode_id}/evaluate", {}
        )

    async def close(self, episode_id: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/v1/episodes/{episode_id}")
