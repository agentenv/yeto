"""TEST workload: a rollout whose every trajectory waits on a (fake) tool call.

For GPU acceptance runs that need real tool waits on the ports path:
A2+ (rl-infra-spec 1.7, tool-wait observation) and A4b (3.3 X5, drain keeps
the old routing while trajectories wait on tools). Plugged in through Miles'
supported per-sample hook ``--custom-generate-function-path
yeto.rl.adapters.miles.harness_glue.tool_wait.generate`` (the ports adapter passes it through;
``--rollout-function-path`` stays adapter-owned).

Each training trajectory first "calls a tool": it sleeps ``delay`` seconds
inside ``tool_wait.async_tool_wait_scope`` on the island's named
``ToolWaitBoard`` actor (so ``MilesRolloutPool.trajectory_load`` counts it,
``--rl-elastic-tool-wait-board``) and adds the wait to
``sample.non_generation_time`` (the 1.7 ``tool_wait_seconds`` source), then
generates with the stock SGLang ``generate``. Eval samples are not delayed.

``delay`` comes from ``YETO_RL_TEST_TOOL_DELAY_S`` (launcher
``--rl-test-tool-delay-s``; the island forwards it to every Ray worker). Unset
or 0: no tool call, the stock generate unchanged.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

TOOL_DELAY_ENV = "YETO_RL_TEST_TOOL_DELAY_S"
# The Miles generate hook moved to the adapter (yeto-framework-decoupling 5.7).
GENERATE_PATH = "yeto.rl.adapters.miles.harness_glue.tool_wait.generate"


def tool_delay_s(environ: Any = None) -> float:
    raw = (os.environ if environ is None else environ).get(TOOL_DELAY_ENV)
    if raw in (None, ""):
        return 0.0
    value = float(raw)
    if value < 0:
        raise ValueError(f"{TOOL_DELAY_ENV} must be >= 0")
    return value


def trajectory_id(sample: Any) -> str:
    return f"g{getattr(sample, 'group_index', None)}-s{getattr(sample, 'index', None)}"


_BOARDS: dict[int, Any] = {}


def _board(learner_id: int) -> Any:
    if learner_id not in _BOARDS:
        from yeto.rl.engine.tool_wait import board_actor

        _BOARDS[learner_id] = board_actor(learner_id)
    return _BOARDS[learner_id]


async def tool_call(sample: Any, *, delay: float, board: Any, sleep=asyncio.sleep,
                    clock=time.monotonic) -> float:
    """The fake tool call: wait ``delay`` s, counted on ``board``; returns the wait."""
    from yeto.rl.engine.tool_wait import async_tool_wait_scope

    started = clock()
    async with async_tool_wait_scope(board, trajectory_id(sample)):
        await sleep(delay)
    waited = clock() - started
    sample.non_generation_time = float(getattr(sample, "non_generation_time", 0.0) or 0.0) + waited
    return waited
