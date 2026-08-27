"""Runtime smoke for the Miles TP4 x CP2 parameter round trip.

Run this with exactly eight ranks in the same node and in the production
Miles/Megatron environment.  It deliberately uses Megatron's runtime process
groups rather than assuming a global-rank layout.
"""

from __future__ import annotations

import math
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

from miles.backends.megatron_utils.update_weight.common import _gather_with_stride


TP_SIZE = 4
CP_SIZE = 2


def inverse_tp_partition(
    full: torch.Tensor,
    *,
    tp_rank: int,
    tp_size: int,
    partition_dim: int,
    partition_stride: int,
) -> torch.Tensor:
    """Invert Miles's ``_gather_with_stride`` exactly."""
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"bad TP rank {tp_rank} for TP size {tp_size}")
    if partition_stride < 1:
        raise ValueError(f"partition_stride must be positive, got {partition_stride}")
    split_count = tp_size * partition_stride
    if full.shape[partition_dim] % split_count:
        raise ValueError(
            f"shape {tuple(full.shape)} dim {partition_dim} is not divisible by "
            f"TP*stride={split_count}"
        )
    chunks = full.chunk(split_count, dim=partition_dim)
    return torch.cat(chunks[tp_rank::tp_size], dim=partition_dim).contiguous()


def _run_case(
    case_id: int,
    full_shape: tuple[int, int],
    partition_dim: int,
    partition_stride: int,
) -> None:
    rank = dist.get_rank()
    tp_group = mpu.get_tensor_model_parallel_group()
    cp_group = mpu.get_context_parallel_group()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()

    full = torch.arange(
        math.prod(full_shape),
        dtype=torch.float32,
        device=torch.cuda.current_device(),
    ).reshape(full_shape)
    full.add_(case_id * 1000)
    expected = inverse_tp_partition(
        full,
        tp_rank=tp_rank,
        tp_size=TP_SIZE,
        partition_dim=partition_dim,
        partition_stride=partition_stride,
    )
    received = torch.empty_like(expected)

    if cp_rank == 0:
        tp_src_global = dist.get_global_rank(tp_group, 0)
        scatter_list = None
        if tp_rank == 0:
            scatter_list = [
                inverse_tp_partition(
                    full,
                    tp_rank=target,
                    tp_size=TP_SIZE,
                    partition_dim=partition_dim,
                    partition_stride=partition_stride,
                )
                for target in range(TP_SIZE)
            ]
        dist.scatter(received, scatter_list=scatter_list, src=tp_src_global, group=tp_group)

        gathered = [torch.empty_like(received) for _ in range(TP_SIZE)]
        dist.all_gather(gathered, received, group=tp_group)
        restored = _gather_with_stride(gathered, partition_dim, partition_stride)
        if not torch.equal(restored, full):
            raise AssertionError(
                f"rank {rank}: TP gather failed for shape={full_shape}, "
                f"dim={partition_dim}, stride={partition_stride}"
            )

    cp_src_global = dist.get_global_rank(cp_group, 0)
    dist.broadcast(received, src=cp_src_global, group=cp_group)
    if not torch.equal(received, expected):
        raise AssertionError(
            f"rank {rank}: scatter/CP broadcast failed for shape={full_shape}, "
            f"dim={partition_dim}, stride={partition_stride}"
        )
    dist.barrier()


def main() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != TP_SIZE * CP_SIZE:
        raise RuntimeError(f"expected {TP_SIZE * CP_SIZE} ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=2))
    try:
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=TP_SIZE,
            pipeline_model_parallel_size=1,
            context_parallel_size=CP_SIZE,
            order="tp-cp-ep-dp-pp",
            create_gloo_process_groups=False,
        )
        tp_group = mpu.get_tensor_model_parallel_group()
        cp_group = mpu.get_context_parallel_group()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        cp_rank = mpu.get_context_parallel_rank()
        if dist.get_rank(tp_group) != tp_rank or dist.get_rank(cp_group) != cp_rank:
            raise AssertionError("Megatron logical ranks disagree with ProcessGroup ranks")

        print(
            {
                "rank": rank,
                "tp_rank": tp_rank,
                "cp_rank": cp_rank,
                "tp_group": dist.get_process_group_ranks(tp_group),
                "cp_group": dist.get_process_group_ranks(cp_group),
            },
            flush=True,
        )

        cases = (
            ((16, 8), 0, 1),
            ((8, 16), 1, 1),
            ((32, 8), 0, 2),
            ((8, 32), 1, 2),
        )
        for case_id, (shape, dim, stride) in enumerate(cases):
            _run_case(case_id, shape, dim, stride)

        if rank == 0:
            print("PASS: TP4 x CP2 scatter, CP broadcast, and stride1/2 gather", flush=True)
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
