"""verl side of the policy-age limit (agentic-rollout-utilization 6.3).

``SUPPORT``: limit 0 runs verl's synchronous trainer (no carry-over, command
line unchanged: :func:`policy_age_overrides` is empty); a limit > 0 runs the
fork's ``experimental/fully_async_policy`` with ``staleness_threshold`` and
``partial_rollout`` derived from the limit; a limit above ``max_policy_age`` is
refused by the launcher before any machine starts.

6.4a (10-09): the translation between verl's fully_async bookkeeping and yeto's
policy-age terms is :mod:`.fully_async_translate` (pure functions, unit tested);
6.4b wires it into the fully_async adapter path (``fully_async_round`` /
``fully_async_ports`` / ``fully_async_runner``): stage 2, limit up to 1.
"""

from __future__ import annotations

from yeto.rl.engine.policy_age import PolicyAgeSupport, validate_limit

# 6.4b (10-09): stage 2 through the fully_async path (rollouter + trainer on
# separate GPUs).  The largest limit is 1 -- the only one the pre-launch review
# (rl-verl-backend 6.6) tests; sub-agent decision, raise after a GPU pass.
SUPPORT = PolicyAgeSupport(backend="verl", stage=2, max_policy_age=1)


def policy_age_overrides(limit: int, *, groups_per_round: int | None = None) -> tuple[str, ...]:
    """Hydra ``async_training`` overrides for the limit: none at 0 (sync trainer,
    command line unchanged); > 0 needs ``groups_per_round`` (one round = one verl
    param version, see ``fully_async_translate.fully_async_overrides``)."""
    if validate_limit(limit) == 0:
        return ()
    SUPPORT.check(limit)
    if groups_per_round is None:
        raise ValueError("policy_age_overrides(limit > 0) needs groups_per_round")
    from .fully_async_translate import fully_async_overrides

    keys = fully_async_overrides(limit, samples_per_round=groups_per_round,
                                 ppo_mini_batch_size=groups_per_round)
    return tuple(f"{k}={v}" for k, v in keys.items())
