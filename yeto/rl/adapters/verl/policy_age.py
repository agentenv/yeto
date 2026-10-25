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

S27 G-EV (VERL-PARITY #10): ``--rl-rollout-ahead 1`` with a limit >= 1 keeps the
synchronous trainer of verl's separate mode (full parameters, trainer and
rollout cards apart: the FSDP2 island with ``--rl-placement fixed-partition``,
or the Megatron island) and lets the rollout servers generate the next batch
while the trainer trains (``VerlRolloutPool.start_generate``). The limit then
bounds that one-version lag; it does not select the fully_async path.
:func:`rollout_ahead_problems` refuses the other verl islands before launch.
"""

from __future__ import annotations

from yeto.rl.engine.policy_age import PolicyAgeSupport, validate_limit

# 6.4b (10-09): stage 2 through the fully_async path (rollouter + trainer on
# separate GPUs).  rl-offpolicy-correction-passthrough L6.9: the cap was 1;
# decoupled PPO needs A_hard >= 2. verl has no upper bound of its own
# (staleness_threshold = N, fully_async_translate.staleness_threshold_for);
# 4 matches the Miles cap. Limits above 1 are not yet run on GPU (L6.10).
SUPPORT = PolicyAgeSupport(backend="verl", stage=2, max_policy_age=4)
# --rl-rollout-ahead: whole batches the separate-mode island generates ahead of its
# trainer (VerlRolloutPool.start_generate; yeto.rl.engine.rollout_ahead).
MAX_ROLLOUT_AHEAD = 1


def rollout_ahead_problems(args) -> list[str]:
    """What a verl launch with ``--rl-rollout-ahead`` > 0 lacks beyond the engine-neutral
    rules (``rollout_ahead.launch_problems``: fixed partition, policy-age limit, no sample
    roles, ...); empty when it can run. Only the full-parameter island in verl's separate
    mode splits its generation; a LoRA island's fixed partition is the fully_async path."""
    from yeto.rl.engine.selection import FULL_PARAMETER_SYNC_PRESET

    if not int(getattr(args, "rl_rollout_ahead", 0) or 0):
        return []
    if (getattr(args, "tuning", "lora") != "full"
            and getattr(args, "rl_sync_preset", None) != FULL_PARAMETER_SYNC_PRESET):
        return ["--tuning full or --rl-sync-preset full-parameter (verl generates ahead only in the "
                "full-parameter separate mode; "
                "a LoRA island's fixed partition is the fully_async path)"]
    return []


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
