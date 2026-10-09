"""TB2 eval attempt: sandbox lease + agent + trusted verifier (rl-eval-difficulty-buckets 3.1).

Composes the reward environment's provider (``tb2_provider.Tb2EnvironmentProvider``;
on Modal ``yeto.cloud.modal_reward_env:modal_provider``) with an agent port:

    agent(lease, task, trial, *, policy_token) -> awaitable mapping
        {"episode_id", "end_reason", "turns", "tokens"}

The verifier runs on the trusted side (``lease.verifier.evaluate``) after the
agent stops; the sandbox is always destroyed. A sandbox that cannot be created
or a verifier error propagates and the island records ``infra_error``; an agent
that ran out of turns / context / time keeps its ``end_reason`` and is still
judged (D6.d: only infrastructure errors are excluded from the pass rate).
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Mapping

Agent = Callable[..., Awaitable[Mapping[str, Any]]]


def trajectory_id(policy_version: int, task_id: str, trial: int) -> str:
    return f"eval-v{int(policy_version)}-{task_id}-t{int(trial)}"


def tb2_attempt(provider: Any, agent: Agent, *, run: Callable[[Awaitable[Any]], Any] = asyncio.run):
    def attempt(task: Any, trial: int, *, policy_version: int, policy_token: str) -> dict[str, Any]:
        tid = trajectory_id(policy_version, task.task_id, trial)

        async def go() -> dict[str, Any]:
            lease = await provider.acquire(task.task_id, tid)
            try:
                out = dict(await agent(lease, task, trial, policy_token=policy_token))
                end = str(out.get("end_reason") or "completed")
                if end == "infra_error":
                    return {**out, "trajectory_id": tid, "reward": 0.0, "success": False}
                verdict = await lease.verifier.evaluate(out.get("episode_id", tid))
            finally:
                await lease.destroy()
            passed = bool(verdict.get("passed"))
            return {"trajectory_id": tid, "reward": 1.0 if passed else 0.0, "success": passed,
                    "end_reason": end, "turns": out.get("turns"), "tokens": out.get("tokens"),
                    "verifier_timed_out": bool(verdict.get("timed_out"))}

        return run(go())

    return attempt
