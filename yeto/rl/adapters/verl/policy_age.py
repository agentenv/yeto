"""verl side of the policy-age limit (agentic-rollout-utilization 6.3).

``SUPPORT``: the verl adapter implements stage 1 (switch and contract) with
limit 0 only; its synchronous trainer has no carry-over. Stage 2 (task 6.4)
derives ``async_training.staleness_threshold`` and ``partial_rollout`` of the
fork's ``experimental/fully_async_policy`` from the limit; until then a limit
> 0 is refused by the launcher before any machine starts, and the verl
command line is unchanged at limit 0 (:func:`policy_age_overrides` is empty).

6.4a (10-09): the translation between verl's fully_async bookkeeping and yeto's
policy-age terms is :mod:`.fully_async_translate` (pure functions, unit tested);
wiring it into a fully_async adapter path is 6.4b, so the declaration stays at
stage 1 and a limit > 0 is still refused before launch.
"""

from __future__ import annotations

from yeto.rl.engine.policy_age import PolicyAgeSupport, validate_limit

SUPPORT = PolicyAgeSupport(backend="verl", stage=1, max_policy_age=0)


def policy_age_overrides(limit: int) -> tuple[str, ...]:
    """Hydra overrides for the limit: none at 0; > 0 is not implemented (stage 2)."""
    if validate_limit(limit) == 0:
        return ()
    SUPPORT.check(limit)  # raises: stage 2 not implemented
    return ()  # pragma: no cover - unreachable until stage 2
