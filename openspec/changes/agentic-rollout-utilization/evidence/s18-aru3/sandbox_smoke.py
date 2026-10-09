"""S18 ARU-3 task 5.3: Modal CPU smoke of the suspended-trajectory sandbox rules.

A (keep-alive): a TB2 sandbox (real Modal, yeto-tbench2 app, the codex provider
   path) runs a command, stays idle for PARK_S seconds (a trajectory suspended
   across a training round: no command runs), then runs a second command that
   reads the first one's file -> state kept, sandbox alive.
B (expiry): a second sandbox; the trajectory's model-turn gate is closed and
   the bridge's SuspendGate waits with a short survival limit -> CodexSuspendExpired;
   the lease is destroyed exactly as ``run``'s finally does -> describe() "gone"
   and Modal reports the sandbox finished.
Cost: sandbox cpu/memory x alive seconds at Modal CPU rates (same constants as
the launch scripts' cost-params: $0.0472/core-h, $0.00000222/GiB-s).

usage: YETO_HARNESS_TB2_TASKS_DIR=... python sandbox_smoke.py OUT_DIR
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from yeto.cloud import modal_reward_env
from yeto.rl.harness.codex import codex_harness_agent as harness
from yeto.rl.harness.codex.environment import HttpTerminalEnvironment

TASK = os.environ.get("SMOKE_TASK", "cancel-async-tasks")
PARK_S = float(os.environ.get("SMOKE_PARK_S", "300"))
EXPIRE_S = float(os.environ.get("SMOKE_EXPIRE_S", "60"))
CORE_H = 0.0472
GIB_S = 0.00000222


def cost(cpus: float, memory_mb: int, seconds: float) -> float:
    return cpus * CORE_H * seconds / 3600 + (memory_mb / 1024) * GIB_S * seconds


async def run_cmd(lease, episode: str, command: str) -> dict:
    async with HttpTerminalEnvironment(lease.env_url, lease.env_token) as env:
        return await env.execute(episode, command, timeout_seconds=60.0, output_bytes=4096)


async def main(out: Path) -> dict:
    provider = modal_reward_env.modal_provider()
    record: dict = {"task": TASK, "park_s": PARK_S, "expire_s": EXPIRE_S,
                    "idle_timeout_s": provider.backend.idle_timeout_s, "ttl_s": provider.backend.ttl_s,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # ---- A: keep-alive across a suspension
    t0 = time.monotonic()
    a = await provider.acquire(TASK, "aru3-smoke-keepalive")
    record["a_acquire_s"] = round(time.monotonic() - t0, 3)
    record["a_sandbox_id"] = getattr(getattr(a.handle, "sandbox", None), "object_id", None)
    record["a_cpus"], record["a_memory_mb"] = a.task.cpus, a.task.memory_mb
    try:
        first = await run_cmd(a, "ep-a", "echo before-suspend > /tmp/aru3 && cat /tmp/aru3")
        record["a_first"] = {k: first.get(k) for k in ("exit_code", "output") if k in first} or str(first)[:300]
        parked = time.monotonic()
        await asyncio.sleep(PARK_S)  # suspended: no command in the sandbox
        record["a_parked_s"] = round(time.monotonic() - parked, 3)
        record["a_describe_after_park"] = await a.describe()
        second = await run_cmd(a, "ep-a", "cat /tmp/aru3")
        record["a_second"] = {k: second.get(k) for k in ("exit_code", "output") if k in second} or str(second)[:300]
        record["a_state_kept"] = "before-suspend" in json.dumps(second)
    finally:
        alive_a = time.monotonic() - t0
        await a.destroy()
        record["a_after_destroy"] = await a.describe()
    record["a_alive_s"] = round(alive_a, 3)
    record["a_cost_usd"] = round(cost(a.task.cpus, a.task.memory_mb, alive_a), 6)
    record["a_parked_cost_usd"] = round(cost(a.task.cpus, a.task.memory_mb, record["a_parked_s"]), 6)

    # ---- B: the survival limit expires -> discarded and released
    t0 = time.monotonic()
    b = await provider.acquire(TASK, "aru3-smoke-expire")
    record["b_sandbox_id"] = getattr(getattr(b.handle, "sandbox", None), "object_id", None)
    gate_path = Path(tempfile.mkdtemp()) / "b.gate"
    gate_path.write_text("suspended\n")
    metrics = harness.legacy.AgentMetrics()
    gate = harness.SuspendGate(str(gate_path), metrics, max_seconds=EXPIRE_S, poll_seconds=0.5)
    try:
        await run_cmd(b, "ep-b", "true")
        try:
            await gate.wait_open()
            record["b_expired"] = False
        except harness.CodexSuspendExpired as exc:
            record["b_expired"] = True
            record["b_reason"] = f"{type(exc).__name__}: {exc}"
        record["b_metrics"] = {k: getattr(metrics, k) for k in harness.SUSPEND_METRIC_FIELDS}
    finally:
        alive_b = time.monotonic() - t0
        await b.destroy()  # run()'s finally
        record["b_after_destroy"] = await b.describe()
    record["b_alive_s"] = round(alive_b, 3)
    record["b_cost_usd"] = round(cost(b.task.cpus, b.task.memory_mb, alive_b), 6)
    record["cost_per_parked_sandbox_minute_usd"] = round(cost(a.task.cpus, a.task.memory_mb, 60.0), 6)
    record["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return record


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(main(out))
    (out / "smoke.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
