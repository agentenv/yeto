"""Miles generate hook of the TEST tool-wait workload (moved from
``yeto/rl/tool_wait_workload.py``, yeto-framework-decoupling 5.7).

The neutral part (delay, fake tool call on the island board) stays in
``yeto.rl.tool_wait_workload``; this wraps Miles' stock SGLang ``generate``.
"""

from __future__ import annotations

from typing import Any

from yeto.rl import tool_wait_workload as twl


async def generate(input: Any) -> Any:  # noqa: A002 - Miles GenerateFnInput protocol
    from miles.rollout.base_types import GenerateFnOutput
    from miles.rollout.sglang_rollout import generate as stock_generate

    delay = twl.tool_delay_s()
    if delay > 0 and not input.evaluation:
        learner_id = int(getattr(input.args, "yeto_rl_learner_id", 0) or 0)
        await twl.tool_call(input.sample, delay=delay, board=twl._board(learner_id))
    sample = await stock_generate(input.args, input.sample, input.sampling_params,
                                  evaluation=input.evaluation)
    return GenerateFnOutput(samples=sample)
