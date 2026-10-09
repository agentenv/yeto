"""Miles side of the policy-age limit (agentic-rollout-utilization 2.1/2.4/4.1).

``SUPPORT``: Miles implements stage 3 (agentic multi-turn carry-over, on top of
stage 2 single-turn carry-over) with a limit of at most 1 (the only value
measured on GPU, design risk "上限先取 1"). :func:`check_task` accepts a limit
> 0 for Miles' stock single-turn generate (stage 2) and for the agentic
generate with an agent function that implements the suspend/resume hooks
(``SUSPEND_CAPABLE_AGENTS``, stage 3); any other custom generate or agent
function is refused, and so is ``--recompute-logprobs-via-prefill`` (it would
overwrite the older version's generation log-probabilities the cross-version
correction needs). Both are checked by the launcher before any machine starts
and again on the island.

:func:`policy_age_argv` derives Miles' own switch from the neutral limit
instead of mapping Miles' boolean ``--partial-rollout`` (design decision 1):
limit 0 emits nothing (default command line unchanged); limit N > 0 emits, for
an agentic run, ``--agentic-suspend-between-turns --agentic-suspend-max-rounds
N`` (agentenv fork: unfinished trajectories are suspended between two model
turns at the cut-off and continued by the next rollout, at most N rollouts
after the one that started them), else ``--partial-rollout`` (unfinished groups go back to Miles' data buffer and
continue next round; :mod:`.carry_over` keeps those within the limit and drops
older ones). Stage 2 no longer emits ``--mask-offpolicy-in-partial-rollout``:
tokens of an older version stay in the loss and are corrected by the TIS the
algorithm spec must configure (``AlgorithmSpec`` refuses staleness > 0 without
a correction), using their generation log-probabilities (design decision 3).
"""

from __future__ import annotations

from yeto.rl.engine.policy_age import STAGE_NAMES, PolicyAgeError, PolicyAgeSupport, validate_limit

SUPPORT = PolicyAgeSupport(backend="miles", stage=3, max_policy_age=1)

PARTIAL_ROLLOUT_FLAGS = ("--partial-rollout",)
AGENTIC_GENERATE = "miles.rollout.generate_hub.agentic_tool_call.generate"
# Agent functions whose module defines the suspend()/resume() hooks Miles calls
# under --agentic-suspend-between-turns (model-turn gate, survival limit).
SUSPEND_CAPABLE_AGENTS = frozenset({
    "yeto.rl.harness.codex.codex_openenv_subprocess_agent_function.run",
})


def policy_age_argv(limit: int, *, agentic: bool = False) -> tuple[str, ...]:
    if validate_limit(limit) == 0:
        return ()
    if agentic:
        return ("--agentic-suspend-between-turns", "--agentic-suspend-max-rounds", str(limit))
    return PARTIAL_ROLLOUT_FLAGS


def check_task(limit: int, *, custom_generate: str | None = None,
               custom_agent: str | None = None, recompute_prefill: bool = False) -> None:
    """Stage 2: single-turn carry-over with Miles' stock generate. Stage 3:
    agentic carry-over with the agentic generate and a suspend-capable agent."""
    if validate_limit(limit) == 0:
        return
    if custom_agent or custom_generate:
        if custom_generate != AGENTIC_GENERATE or custom_agent not in SUSPEND_CAPABLE_AGENTS:
            used = [f"{flag} {value}" for flag, value in (
                ("--custom-generate-function-path", custom_generate),
                ("--custom-agent-function-path", custom_agent),
            ) if value]
            raise PolicyAgeError(
                f"--rl-max-policy-age {limit} with {', '.join(used)}: backend 'miles' supports "
                f"carry-over for single-turn tasks with Miles' stock generate (stage 2, "
                f"{STAGE_NAMES[2]}) and for agentic tasks with {AGENTIC_GENERATE} and an agent "
                f"function that can be suspended between turns ({', '.join(sorted(SUSPEND_CAPABLE_AGENTS))}; "
                f"stage 3, {STAGE_NAMES[3]}); refused before any machine starts")
    if recompute_prefill:
        raise PolicyAgeError(
            f"--rl-max-policy-age {limit} with --recompute-logprobs-via-prefill: the prefill "
            "recompute overwrites the older versions' generation log-probabilities that the "
            "cross-version importance-sampling correction needs")
