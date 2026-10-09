"""verl adapter entry role for the launcher (rl-verl-backend; decoupling stage W role).

``verl_capabilities`` is what the verl island declares (first step): LoRA,
colocated one-GPU islands, GRPO, corrections none / TIS (lower bound 0).  The
launcher looks the function up under the Miles-era name ``miles_capabilities``
(kept as an alias here until the launcher's call site is renamed).
"""

from __future__ import annotations

POLICY_TOKEN_SOURCE = ("publication (sync colocated by construction), not a per-sample engine report")


def verl_capabilities(fingerprint: str, *, unverified_mechanisms=()):
    from yeto.rl.engine.capabilities import EngineCapabilities, ExecutionCapabilities

    from .policy_age import SUPPORT

    capabilities = EngineCapabilities(
        engine="verl",
        runtime_fingerprint=fingerprint,
        parameter_layouts={"lora"},
        placements={"colocated"},
        advantage_estimators={"grpo"},
        dynamic_sampling_filters=set(),
        execution_modes={"colocated-serial"},
        corrections={"none", "tis"},
        # agentic-rollout-utilization 6.3: declared from the policy-age support
        # (stage 1: limit 0) instead of a hard-coded 0.
        execution=ExecutionCapabilities(critic=False, max_policy_staleness=SUPPORT.max_policy_age,
                                        rollout_logprobs=True),
        extra={"policy_token_source": POLICY_TOKEN_SOURCE},
    )
    if unverified_mechanisms:
        capabilities = capabilities.with_unverified(unverified_mechanisms)
    return capabilities


miles_capabilities = verl_capabilities  # launcher call-site name (decoupling rename pending)


def with_partitioned_serial(capabilities):
    raise ValueError("verl 后端第一版不支持 fixed-partition 放置")


def connect_island_ray(*_args, **_kwargs):
    raise NotImplementedError("verl 岛自己在 verl_main 里连接 Ray（stage W 不适用）")
