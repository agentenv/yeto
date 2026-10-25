"""verl rollout role (``rollout``; S27 G-DC, V10).

The rollout ports of the verl islands: :class:`~.ports_impl.VerlRolloutPool`
for both engines (the FSDP2 colocated island and the Megatron island with its
standalone vLLM servers). The Miles rollout role also carries the elastic test
hook ``INJECT_START_DELAY_ENV``; no verl island reads such a hook, so asking
for it is an error, not a silently ignored variable.
"""

from __future__ import annotations

ROLLOUT_ENGINE = "vllm"


def rollout_pool_class():
    from .ports_impl import VerlRolloutPool

    return VerlRolloutPool


def __getattr__(name: str):
    if name == "INJECT_START_DELAY_ENV":
        raise ValueError("verl 后端没有接 --rl-test-inject-start-delay-s（rollout 启动延迟注入只在 Miles 岛上）")
    raise AttributeError(name)
