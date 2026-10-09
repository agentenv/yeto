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


def verl_capabilities(fingerprint: str, *, unverified_mechanisms=(), fully_async_limit: int = 0):
    """``fully_async_limit`` 0: the sync colocated trainer (age 0).  > 0 (6.4b): the
    fully_async path -- rollouter and trainer on separate GPUs (fixed partition,
    driver still serial), policy age up to the limit (checked against SUPPORT)."""
    from yeto.rl.engine.capabilities import EngineCapabilities, ExecutionCapabilities

    from .policy_age import SUPPORT

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
    raise ValueError("verl 后端第一版不支持 fixed-partition 放置")


def connect_island_ray(*_args, **_kwargs):
    raise NotImplementedError("verl 岛自己在 verl_main 里连接 Ray（stage W 不适用）")
