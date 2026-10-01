"""Trusted-side entry for the Terminal-Bench Codex adapter (REWRITE).

``run`` is the ``--custom-agent-function-path`` target under upstream
``agentic_tool_call.generate``.  It owns the environment lease, spawns the
untrusted worker with the reward key scrubbed from its environment, maps the
worker's tool-wait events onto the ToolWaitBoard (3.1), runs the verifier and
signs the outcome (D7), and guarantees cleanup on cancellation/timeouts:
worker process group, environment lease, tool-wait scope.

Environment provisioning is injected (``EnvironmentProvider``) so the sandbox
backend (D9) and the CPU fake share one code path.  Nothing here touches
INFRA-owned files; the ToolWaitBoard handle is passed in by the caller/pool
(IR stub: see CODEX-PROGRESS "等 IR 合入后要替换的桩").
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from yeto.rl.engine.tool_wait import _call as _board_call

from . import codex_openenv_agent_function as adapter
from .environment import TerminalEnvironment, TrustedVerifier

WORKER_MODULE = "yeto.rl.harness.codex.codex_openenv_agent_worker"
WORKER_GRACE_SECONDS = 5.0


@dataclass
class EnvironmentLease:
    """One environment bound to one trajectory (R-D9: deadline is a hard cap)."""

    env_url: str
    env_token: str
    verifier: TrustedVerifier
    destroy: Callable[[], Awaitable[None]]
    describe: Callable[[], Awaitable[str]]  # "live" | "gone"
    deadline_seconds: float


class EnvironmentProvider(Protocol):
    async def acquire(self, task_id: str, trajectory_id: str) -> EnvironmentLease: ...


_provider: EnvironmentProvider | None = None
_tool_wait_board: Any = None


def configure(*, provider: EnvironmentProvider | None, tool_wait_board: Any = None) -> None:
    """Install the environment provider and (optional) ToolWaitBoard handle."""
    global _provider, _tool_wait_board
    _provider = provider
    _tool_wait_board = tool_wait_board


def scrubbed_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for name in adapter.hmac_key_env_names():
        env.pop(name, None)
    return env


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    for sig, wait in ((signal.SIGTERM, WORKER_GRACE_SECONDS), (signal.SIGKILL, WORKER_GRACE_SECONDS)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        # asyncio.wait (not wait_for): must work inside a cancelled task's finally.
        waiter = asyncio.ensure_future(process.wait())
        done, _pending = await asyncio.wait({waiter}, timeout=wait)
        if done:
            return
        waiter.cancel()


async def _drive_worker(job: dict[str, Any], trajectory_id: str, board: Any) -> dict[str, Any]:
    """Spawn the worker, relay tool-wait events, return its result event."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        WORKER_MODULE,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL if os.getenv("YETO_CODEX_WORKER_STDERR") != "1" else None,
        env=scrubbed_environment(),
        start_new_session=True,
    )
    in_tool = False
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write((json.dumps(job) + "\n").encode())
        await process.stdin.drain()
        process.stdin.close()
        while True:
            line = await process.stdout.readline()
            if not line:
                raise adapter.harness.CodexHarnessError("worker exited without a result")
            event = json.loads(line)
            kind = event.get("event")
            if kind == "tool_wait":
                if event.get("phase") == "enter" and not in_tool:
                    in_tool = True
                    if board is not None:
                        _board_call(board, "enter", trajectory_id)
                elif event.get("phase") == "exit" and in_tool:
                    in_tool = False
                    if board is not None:
                        _board_call(board, "exit", trajectory_id)
            elif kind == "result":
                return event
            elif kind == "error":
                raise adapter.harness.CodexHarnessError(str(event.get("reason")))
    finally:
        if in_tool and board is not None:
            _board_call(board, "exit", trajectory_id)
        await _terminate(process)


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any] | None:
    metadata = dict(metadata or {})
    if _provider is None:
        raise RuntimeError("codex_openenv_subprocess_agent_function.configure(provider=...) was not called")
    task_id, sample_id = adapter.task_identity(metadata)
    trajectory_id = str(metadata.get("trajectory_id") or sample_id)
    episode_id = adapter.new_episode_id()
    fields = adapter.trajectory_fields(trajectory_id)
    expected_version = metadata.get("expected_policy_version")
    lease: EnvironmentLease | None = None
    try:
        try:
            lease = await _provider.acquire(task_id, trajectory_id)
        except Exception as exc:  # noqa: BLE001 - provisioning failures are infrastructure
            return {**adapter.infrastructure_metadata(f"acquire: {type(exc).__name__}: {exc}", episode_id=None), **fields}
        job = {
            "base_url": base_url,
            "prompt": metadata.get("prompt") if isinstance(metadata.get("prompt"), str) else str(prompt),
            "request_kwargs": dict(request_kwargs or {}),
            "episode_id": episode_id,
            "max_seq_len": metadata.get("max_seq_len"),
            "env_url": lease.env_url,
            "env_token": lease.env_token,
            "driver": metadata.get("codex_openenv_driver", "stock"),
            **{k: metadata[k] for k in ("script", "final_status", "hang_seconds") if k in metadata},
        }
        try:
            untrusted = await asyncio.wait_for(
                _drive_worker(job, trajectory_id, _tool_wait_board), timeout=lease.deadline_seconds
            )
        except asyncio.TimeoutError:
            untrusted = {"status": "timeout", "metrics": {"timed_out": 1}, "episode_id": episode_id}
        except adapter.harness.CodexHarnessError as exc:
            return {**adapter.infrastructure_metadata(str(exc), episode_id=episode_id), **fields}
        signed = await adapter.finish_trusted(untrusted, lease.verifier, task_id=task_id, sample_id=sample_id)
        if expected_version is not None:
            signed["expected_policy_version"] = expected_version
        return {**signed, **fields}
    finally:
        if lease is not None:
            await lease.destroy()
            if await lease.describe() != "gone":
                raise RuntimeError(f"environment for {trajectory_id} was not confirmed destroyed")
