"""Untrusted worker process for the Terminal-Bench Codex adapter (REWRITE).

Protocol (stdin/stdout, one JSON object per line):
- stdin: one job ``{"base_url", "prompt", "request_kwargs", "episode_id",
  "max_seq_len", "env_url", "env_token", "driver": "stock"|"scripted", ...}``;
- stdout events: ``{"event": "tool_wait", "phase": "enter"|"exit"}`` for every
  tool execution, then exactly one ``{"event": "result", ...}`` or
  ``{"event": "error", "reason": ...}``.

The worker refuses to start when a reward HMAC key is visible (trust split).
``driver: "scripted"`` is a CPU-test driver (no Codex, no Miles) and is only
honoured when ``YETO_CODEX_OPENENV_ALLOW_SCRIPTED_DRIVER=1``; preflight fails
closed when that variable is set.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from . import codex_openenv_agent_function as adapter
from .environment import HttpTerminalEnvironment

SCRIPTED_DRIVER_ENV = "YETO_CODEX_OPENENV_ALLOW_SCRIPTED_DRIVER"


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


async def _scripted_drive(job: dict[str, Any], env: Any, tool_wait: adapter.ToolWaitEmitter) -> dict[str, Any]:
    """Execute ``job['script']`` commands then submit; mimics the driver's tool loop."""
    scoped = adapter._ToolWaitEnvironment(env, tool_wait)
    for command in job.get("script") or []:
        await scoped.execute(job["episode_id"], str(command), timeout_seconds=30.0, output_bytes=4096)
        if job.get("hang_seconds"):
            await asyncio.sleep(float(job["hang_seconds"]))
    await scoped.submit(job["episode_id"], {"evidence": "scripted"})
    return {"status": job.get("final_status", "completed"), "metrics": {"turns": len(job.get("script") or []) + 1}, "episode_id": job["episode_id"]}


async def _main_async(job: dict[str, Any]) -> dict[str, Any]:
    tool_wait = adapter.ToolWaitEmitter(
        enter=lambda: _emit({"event": "tool_wait", "phase": "enter"}),
        exit=lambda: _emit({"event": "tool_wait", "phase": "exit"}),
    )
    async with HttpTerminalEnvironment(job["env_url"], job["env_token"]) as env:
        if job.get("driver") == "scripted":
            if os.getenv(SCRIPTED_DRIVER_ENV) != "1":
                raise RuntimeError("scripted driver is not allowed in this process")
            return await _scripted_drive(job, env, tool_wait)
        return await adapter.drive_untrusted(job, env, tool_wait=tool_wait)


def main(argv: list[str] | None = None) -> int:
    del argv
    try:
        adapter.assert_no_reward_key()
        job = json.loads(sys.stdin.readline())
        if not isinstance(job, dict):
            raise ValueError("worker job must be a JSON object")
        result = asyncio.run(_main_async(job))
    except BaseException as exc:  # noqa: BLE001 - every failure is reported on the wire
        event = {"event": "error", "reason": f"{type(exc).__name__}: {exc}"}
        metrics = getattr(exc, "metrics", None)
        if isinstance(metrics, dict):
            event["metrics"] = metrics  # G6a: rejection counters reach the trusted layer
        _emit(event)
        return 1
    _emit({"event": "result", **result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
