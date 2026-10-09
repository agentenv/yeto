"""verl-only checks of the neutral run config (decoupling 4.2 role; rl-verl-backend).

First step: one GPU per island, FSDP2, no Megatron reference checkpoint.
"""

from __future__ import annotations


def resolve_ref_load(args, model_path) -> str:
    if getattr(args, "megatron_ref_load", None) is not None:
        raise ValueError("--megatron-ref-load is a Miles/Megatron option; verl uses the HF model")
    return str(model_path)


def check_trainer_parallel(actor_gpus: int, tensor_parallel: int, pipeline_parallel: int) -> None:
    if (tensor_parallel, pipeline_parallel) != (1, 1):
        raise ValueError("verl FSDP2 backend: tensor/pipeline parallel must be 1")
    if actor_gpus != 1:
        raise ValueError("verl backend first step: one GPU per island")


def check_expert_parallel(actor_gpus: int, expert_parallel: int) -> None:
    if expert_parallel != 1:
        raise ValueError("verl backend first step: no expert parallelism")


def check_global_batch(global_batch: int, data_parallel: int) -> None:
    if data_parallel != 1 or global_batch <= 0:
        raise ValueError("verl backend first step: data parallel 1 and a positive batch")


def check_over_sampling(rollout_batch_size: int, over_sampling_batch_size: int) -> None:
    """agentic-rollout-utilization 6.2: the pinned verl fork has no usable
    over-sample cut-off in synchronous mode (``rollout.over_sample_rate`` is
    defined but never used), so over-sampling is refused, not ignored."""
    if over_sampling_batch_size != rollout_batch_size:
        raise ValueError(
            "verl 后端尚未实现多发截止（阶段 0，任务 6.2b）：--over-sampling-batch-size "
            f"{over_sampling_batch_size} 必须等于 --rollout-batch-size {rollout_batch_size}")
