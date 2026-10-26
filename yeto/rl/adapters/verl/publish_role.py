"""verl publication role (``publish`` in the backend registry; S27 G-DC, V10).

(``publish.py`` of this adapter is the LoRA publication checksum module.)

How a verl island puts a policy into its rollout engines:

* FSDP2 island (LoRA): :class:`~.ports_impl.VerlPublisher` -- colocated
  in-memory update plus the read-back checksum of the registered adapter;
* Megatron island (full parameters): :class:`~.megatron_ports.MegatronPublisher`
  -- verl's NCCL checkpoint engine to the standalone vLLM servers; no engine
  read-back (the per-round train/rollout log-prob criteria are the check).

The Miles role also carries the elastic test hook ``INJECT_UPDATE_BLOCK_ENV``;
no verl island reads it, so asking for it is an error.
"""

from __future__ import annotations


def publisher_class(engine: str):
    if engine == "megatron":
        from .megatron_ports import MegatronPublisher

        return MegatronPublisher
    from .ports_impl import VerlPublisher

    return VerlPublisher


def __getattr__(name: str):
    if name == "INJECT_UPDATE_BLOCK_ENV":
        raise ValueError("verl 后端没有接 --rl-test-inject-update-weights-block-s（只在 Miles 岛上）")
    raise AttributeError(name)
