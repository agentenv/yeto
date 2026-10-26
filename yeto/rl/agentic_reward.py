"""The agentic path's reward comes from the agent harness, not from a grader here (S27 G-FS).

A Harbor trial's verdict is produced by the task's own tests inside the sandbox
and is already on the sample before any reward function could run:

* verl: ``HarborAgentLoop`` returns it as ``AgentLoopOutput.reward_score``, which
  verl writes into ``rm_scores``; its reward manager then returns that tensor and
  never calls a custom reward function (fork acad9875,
  ``workers/reward_manager/batch.py:82-84``).
* Miles: ``adapters/miles/harbor_agent.reward_func`` reads the same verdict.

The launcher still requires a ``--reward-function``, so an agentic verl run names
``yeto.rl.agentic_reward:score``.  Both entry points raise: a call means the
verdict did not reach the sample, and a silent 0.0 would train on nothing while
looking healthy.  ``reward_fn.compute_score_agentic`` is the verl spelling of the
same assertion.
"""

from __future__ import annotations

__all__ = ["score"]

MESSAGE = ("the agentic path scores with the Harbor verdict on the sample "
           "(verl: AgentLoopOutput.reward_score -> rm_scores); a reward function was called instead, "
           "so the verdict did not reach the sample")


async def score(args, sample, **kwargs):
    raise AssertionError(MESSAGE)
