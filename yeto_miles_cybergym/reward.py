"""Execution-grounded CyberGym reward callable for the Miles RM contract.

Miles calls a custom reward as ``async_rm(args, sample)`` and can optionally
call it in batched mode as ``async_rm(args, samples)``.  Yeto's strict RL
launcher also uses the same source as a batch reward function.  This module
keeps the network protocol in one place and deliberately raises on transport
or server errors: an unavailable CyberGym runner is not a negative example.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_SERVER_URL = "http://127.0.0.1:8666"
DEFAULT_AGENT_ID = "yeto_agent"
DEFAULT_SALT = "CyberGym"
DEFAULT_TIMEOUT = 60.0

_SUBMISSION_LOCKS: dict[tuple[str, str, str, bytes], threading.Lock] = {}
_SUBMISSION_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class CyberGymConfig:
    """Connection and checksum settings read by each worker process."""

    server_url: str = DEFAULT_SERVER_URL
    agent_id: str = DEFAULT_AGENT_ID
    salt: str = DEFAULT_SALT
    api_key: str = ""
    timeout: float = DEFAULT_TIMEOUT

    @classmethod
    def from_env(cls) -> "CyberGymConfig":
        timeout_text = os.environ.get("CYBERGYM_TIMEOUT", str(DEFAULT_TIMEOUT))
        try:
            timeout = float(timeout_text)
        except ValueError as exc:
            raise ValueError("CYBERGYM_TIMEOUT must be a positive number") from exc
        if timeout <= 0:
            raise ValueError("CYBERGYM_TIMEOUT must be a positive number")
        return cls(
            server_url=os.environ.get("CYBERGYM_URL", DEFAULT_SERVER_URL).rstrip("/"),
            agent_id=os.environ.get("CYBERGYM_AGENT_ID", DEFAULT_AGENT_ID),
            salt=os.environ.get("CYBERGYM_SALT", DEFAULT_SALT),
            api_key=os.environ.get("CYBERGYM_API_KEY", ""),
            timeout=timeout,
        )


def compute_checksum(task_id: str, agent_id: str, salt: str = DEFAULT_SALT) -> str:
    """Return the checksum required by CyberGym's ``submit-vul`` endpoint."""

    return hashlib.sha256(f"{task_id}{agent_id}{salt}".encode("utf-8")).hexdigest()


def compute_reward(exit_code: int | None) -> float:
    """Map CyberGym's vulnerable-runner exit code to the smoke reward."""

    return -1.0 if exit_code in (0, 300, -1, None) else 1.0


def _field(sample: Any, name: str) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(name)
    return getattr(sample, name, None)


def _metadata(sample: Any) -> Mapping[str, Any]:
    metadata = _field(sample, "metadata")
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("CyberGym sample metadata must be a mapping")
    return metadata


def _response_bytes(sample: Any) -> bytes:
    response = _field(sample, "response")
    if response is None:
        response = _field(sample, "completion")
    if isinstance(response, bytes):
        return response
    if isinstance(response, str):
        return response.encode("utf-8")
    if response is None:
        raise ValueError("CyberGym sample has no response to submit")
    return str(response).encode("utf-8")


def _submission_lock(
    config: CyberGymConfig, task_id: str, poc_bytes: bytes
) -> threading.Lock:
    key = (
        config.server_url,
        config.agent_id,
        task_id,
        hashlib.sha256(poc_bytes).digest(),
    )
    with _SUBMISSION_LOCKS_GUARD:
        lock = _SUBMISSION_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SUBMISSION_LOCKS[key] = lock
        return lock


def _submit_one(sample: Any, config: CyberGymConfig) -> float:
    metadata = _metadata(sample)
    task_id = metadata.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("CyberGym sample metadata must contain a non-empty task_id")

    submission_metadata = {
        "agent_id": config.agent_id,
        "task_id": task_id,
        "checksum": compute_checksum(task_id, config.agent_id, config.salt),
        "require_flag": bool(metadata.get("require_flag", False)),
    }
    poc_bytes = _response_bytes(sample)
    files = {
        "metadata": (None, json.dumps(submission_metadata), "application/json"),
        "file": ("poc", poc_bytes, "application/octet-stream"),
    }
    headers = {"X-API-Key": config.api_key} if config.api_key else {}
    try:
        with _submission_lock(config, task_id, poc_bytes):
            response = requests.post(
                f"{config.server_url}/submit-vul",
                files=files,
                headers=headers,
                timeout=config.timeout,
            )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"CyberGym submission failed: {exc}") from exc

    if response.status_code != 200:
        detail = getattr(response, "text", "")[:500]
        raise RuntimeError(
            f"CyberGym submission returned HTTP {response.status_code}: {detail}"
        )
    try:
        result = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("CyberGym returned a non-JSON success response") from exc
    if not isinstance(result, Mapping):
        raise RuntimeError("CyberGym returned a JSON value instead of an object")
    return compute_reward(result.get("exit_code", -1))


async def score(args: Any, samples: Any, **_: Any) -> float | list[float]:
    """Submit one Miles sample or a sequence of samples and return rewards.

    Requests run in worker threads so the async Miles rollout loop is not
    blocked while CyberGym starts its Docker runner.
    """

    config = CyberGymConfig.from_env()
    if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes, bytearray)):
        return [
            await asyncio.to_thread(_submit_one, sample, config)
            for sample in samples
        ]
    return await asyncio.to_thread(_submit_one, samples, config)
