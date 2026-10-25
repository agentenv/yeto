"""verl adapter entry role for the launcher (rl-verl-backend; decoupling stage W role).

``verl_capabilities`` is what the verl island declares (first step): LoRA,
colocated one-GPU islands, GRPO, corrections none / TIS (lower bound 0).  The
launcher looks the function up under the Miles-era name ``miles_capabilities``
(kept as an alias here until the launcher's call site is renamed).
"""

from __future__ import annotations

POLICY_TOKEN_SOURCE = ("publication (sync colocated by construction), not a per-sample engine report")
FULLY_ASYNC_TOKEN_SOURCE = ("per-trajectory min/max_global_steps + per-call versions (yeto build patch), "
                            "mapped to outer versions by the publication VersionMap")


MEGATRON_FEATURES = ("eps_clip", "clip_higher")
MEGATRON_TOKEN_SOURCE = ("publication (separate trainer/rollout pools, prompts submitted after the "
                         "verified publication), not a per-sample engine report")


def verl_capabilities(fingerprint: str, *, unverified_mechanisms=(), fully_async_limit: int = 0,
                      megatron: bool = False):
    """``fully_async_limit`` 0: the sync colocated trainer (age 0).  > 0 (6.4b): the
    fully_async path -- rollouter and trainer on separate GPUs (fixed partition,
    driver still serial), policy age up to the limit (checked against SUPPORT).

    ``megatron`` (S27 G-DC): the Megatron island -- full parameters kept in the
    trainer (``fragments`` layout), trainer and rollout pools apart (fixed
    partition, driver serial), age 0."""
    from yeto.rl.engine.capabilities import EngineCapabilities, ExecutionCapabilities

    from .policy_age import SUPPORT

    if megatron:
        if fully_async_limit:
            raise ValueError("verl Megatron island: the fully_async path is not wired")
        capabilities = EngineCapabilities(
            engine="verl", runtime_fingerprint=fingerprint, parameter_layouts={"fragments"},
            placements={"fixed-partition"}, advantage_estimators={"grpo"}, dynamic_sampling_filters=set(),
            execution_modes={"partitioned-serial"}, corrections={"none", "tis"},
            # mapped to clip_ratio_low / clip_ratio_high by megatron_engine.build_overrides
            features=frozenset(MEGATRON_FEATURES),
            execution=ExecutionCapabilities(critic=False, max_policy_staleness=0, rollout_logprobs=True),
            extra={"policy_token_source": MEGATRON_TOKEN_SOURCE, "train_engine": "megatron"})
        return capabilities.with_unverified(unverified_mechanisms) if unverified_mechanisms else capabilities
    if fully_async_limit:
        SUPPORT.check(fully_async_limit)
    capabilities = EngineCapabilities(
        engine="verl",
        runtime_fingerprint=fingerprint,
        parameter_layouts={"lora"},
        placements={"fixed-partition"} if fully_async_limit else {"colocated"},
        advantage_estimators={"grpo"},
        dynamic_sampling_filters=set(),
        execution_modes={"partitioned-serial"} if fully_async_limit else {"colocated-serial"},
        corrections={"none", "tis"},
        # agentic-rollout-utilization 6.3/6.4b: the sync path produces age 0; the
        # fully_async path up to its limit (<= SUPPORT.max_policy_age).
        execution=ExecutionCapabilities(critic=False, max_policy_staleness=int(fully_async_limit),
                                        rollout_logprobs=True),
        extra={"policy_token_source": FULLY_ASYNC_TOKEN_SOURCE if fully_async_limit
               else POLICY_TOKEN_SOURCE},
    )
    if unverified_mechanisms:
        capabilities = capabilities.with_unverified(unverified_mechanisms)
    return capabilities


miles_capabilities = verl_capabilities  # launcher call-site name (decoupling rename pending)


def with_partitioned_serial(capabilities):
    """Launcher-side declaration of the fixed-partition placement (S27 G-DC): the verl
    Megatron island (trainer and rollout pools apart, driver serial) and the
    fully_async path run it. No partitioned-overlap mode on verl."""
    import dataclasses

    return dataclasses.replace(
        capabilities, placements=capabilities.placements | {"fixed-partition"},
        execution_modes=capabilities.execution_modes | {"partitioned-serial"})


def connect_island_ray(*_args, **_kwargs):
    raise NotImplementedError("verl 岛自己在 verl_main 里连接 Ray（stage W 不适用）")
