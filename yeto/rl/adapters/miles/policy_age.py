"""Miles side of the policy-age limit (agentic-rollout-utilization 2.1/2.4/4.1).

``SUPPORT``: Miles implements stage 2 (single-turn carry-over) with a limit of
at most 1 (the only value measured on GPU, design risk "上限先取 1"). Stage 2
is single-turn only: :func:`check_task` refuses a limit > 0 with an agent
function or a custom generate function (agentic multi-turn carry-over is
stage 3) and with ``--recompute-logprobs-via-prefill`` (it would overwrite the
older version's generation log-probabilities the cross-version correction
needs). Both are checked by the launcher before any machine starts and again
on the island.

:func:`policy_age_argv` derives Miles' own switch from the neutral limit
instead of mapping Miles' boolean ``--partial-rollout`` (design decision 1):
limit 0 emits nothing (default command line unchanged); limit N > 0 emits
``--partial-rollout`` (unfinished groups go back to Miles' data buffer and
continue next round; :mod:`.carry_over` keeps those within the limit and drops
older ones). Stage 2 no longer emits ``--mask-offpolicy-in-partial-rollout``:
tokens of an older version stay in the loss and are corrected by the TIS the
algorithm spec must configure (``AlgorithmSpec`` refuses staleness > 0 without
a correction), using their generation log-probabilities (design decision 3).
"""

from __future__ import annotations

from yeto.rl.engine.policy_age import STAGE_NAMES, PolicyAgeError, PolicyAgeSupport, validate_limit

SUPPORT = PolicyAgeSupport(backend="miles", stage=2, max_policy_age=1)

PARTIAL_ROLLOUT_FLAGS = ("--partial-rollout",)


def policy_age_argv(limit: int) -> tuple[str, ...]:
    return PARTIAL_ROLLOUT_FLAGS if validate_limit(limit) > 0 else ()


def check_task(limit: int, *, custom_generate: str | None = None,
               custom_agent: str | None = None, recompute_prefill: bool = False) -> None:
    """Stage 2 is single-turn carry-over with Miles' stock generate only."""
    if validate_limit(limit) == 0:
        return
    used = [flag for flag, value in (
        ("--custom-agent-function-path", custom_agent),
        ("--custom-generate-function-path", custom_generate),
    ) if value]
    if used:
        raise PolicyAgeError(
            f"--rl-max-policy-age {limit} with {', '.join(used)}: backend 'miles' supports "
            f"carry-over only for single-turn tasks with Miles' stock generate (stage 2, "
            f"{STAGE_NAMES[2]}); agentic multi-turn carry-over is stage 3 ({STAGE_NAMES[3]}) "
            "and is refused before any machine starts")
    if recompute_prefill:
        raise PolicyAgeError(
            f"--rl-max-policy-age {limit} with --recompute-logprobs-via-prefill: the prefill "
            "recompute overwrites the older versions' generation log-probabilities that the "
            "cross-version importance-sampling correction needs")
