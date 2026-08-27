"""Pure-Torch state transforms for the Miles value-training integration.

This module deliberately has no Megatron-Core or Transformer Engine imports.
Callers resolve model-parallel groups and optimizer wrappers in the Miles
runtime, then pass ordinary :class:`torch.Tensor` objects to these helpers.

The BF16/remainder representation below follows Transformer Engine v2.17's
``transformer_engine/common/multi_tensor/adam.cu`` exactly.  In particular,
it is *not* interchangeable with a normal ``float32 -> bfloat16`` cast: TE
rounds the stored high 16 bits upward whenever the low 16-bit word is negative
when interpreted as ``int16_t``.  The same predicate is used to undo that
rounding before reconstructing the FP32 master value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch


MasterParamStorageKind = Literal["fp32_master", "bf16_remainder"]


@dataclass(frozen=True)
class MasterParamStorage:
    """The optimizer-owned tensor that represents an FP32 master parameter.

    ``tensor`` is either a contiguous FP32 master tensor or TE's contiguous
    INT16 low-word remainder tensor.  Selecting it is read-only: no optimizer
    state, including ``exp_avg`` and ``exp_avg_sq``, is changed.
    """

    kind: MasterParamStorageKind
    tensor: torch.Tensor


def _require_plain_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _require_dense_contiguous(name: str, tensor: object) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must have torch.strided layout, got {tensor.layout}")
    if tensor.device.type == "meta":
        raise ValueError(f"{name} must contain materialized data, not a meta tensor")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def normalize_miles_partition(
    name: str,
    *,
    swiglu: bool,
    partition_stride: int,
    partition_dim: int,
) -> tuple[int, int]:
    """Apply Miles's Qwen partition metadata corrections, fail-closed.

    The return order intentionally matches Miles
    ``_check_and_fix_partition``: ``(partition_stride, partition_dim)``.

    Miles forces SwiGLU ``linear_fc1.weight`` to stride 2, and corrects a
    historically misreported ``linear_fc2.weight`` partition dimension from
    0 to 1.  All other currently supported dense parameters must have stride
    1.  Unknown stride values are rejected instead of being guessed.
    """

    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty str")
    if type(swiglu) is not bool:
        raise TypeError(f"swiglu must be a bool, got {type(swiglu).__name__}")
    stride = _require_plain_int("partition_stride", partition_stride)
    dim = _require_plain_int("partition_dim", partition_dim)

    if "linear_fc1.weight" in name and swiglu:
        # Older metadata can report 1; current Megatron reports 2.  Miles
        # normalizes both to the authoritative fused gate/up stride of 2.
        if stride not in (1, 2):
            raise ValueError(
                f"unsupported partition_stride={stride} for SwiGLU parameter {name!r}"
            )
        stride = 2
    elif "linear_fc2.weight" in name:
        if stride != 1:
            raise ValueError(
                f"expected partition_stride=1 for {name!r}, got {stride}"
            )
        if dim == 0:
            dim = 1
    elif stride != 1:
        raise ValueError(
            f"unsupported partition_stride={stride} for parameter {name!r}; "
            "only SwiGLU linear_fc1.weight may use stride 2"
        )

    return stride, dim


def inverse_tp_partition(
    full: torch.Tensor,
    *,
    tp_rank: int,
    tp_size: int,
    partition_dim: int,
    partition_stride: int,
) -> torch.Tensor:
    """Return one TP shard that exactly inverts Miles ``_gather_with_stride``.

    If the canonical tensor is split into ``tp_size * partition_stride``
    equal chunks ``C``, TP rank ``r`` receives::

        cat(C[r], C[tp_size + r], C[2 * tp_size + r], ...)

    Miles's gather splits each local shard back into ``partition_stride``
    pieces and concatenates with stride outermost and TP rank innermost, which
    recovers the canonical tensor element-for-element.  Strides other than 1
    and the Qwen/SwiGLU stride 2 are rejected because Miles does not currently
    authorize them for dense value-model parameters.
    """

    full = _require_dense_contiguous("full", full)
    rank = _require_plain_int("tp_rank", tp_rank)
    size = _require_plain_int("tp_size", tp_size)
    dim = _require_plain_int("partition_dim", partition_dim)
    stride = _require_plain_int("partition_stride", partition_stride)

    if full.ndim == 0:
        raise ValueError("full must have at least one dimension")
    if size <= 0:
        raise ValueError(f"tp_size must be positive, got {size}")
    if rank < 0 or rank >= size:
        raise ValueError(f"tp_rank={rank} is outside [0, {size})")
    if stride not in (1, 2):
        raise ValueError(
            f"partition_stride must be 1 or 2 for Miles dense parameters, got {stride}"
        )
    if dim < -full.ndim or dim >= full.ndim:
        raise ValueError(
            f"partition_dim={dim} is invalid for a rank-{full.ndim} tensor"
        )
    dim %= full.ndim

    split_count = size * stride
    dim_size = full.shape[dim]
    if dim_size == 0 or dim_size % split_count != 0:
        raise ValueError(
            f"full shape {tuple(full.shape)} along dim {dim} must be non-zero and "
            f"divisible by tp_size * partition_stride = {split_count}"
        )

    chunks = full.chunk(split_count, dim=dim)
    if len(chunks) != split_count:
        # The divisibility check above should make this unreachable.  Keep the
        # guard so a changed torch.chunk contract cannot silently corrupt TP.
        raise RuntimeError(
            f"torch.chunk returned {len(chunks)} chunks, expected {split_count}"
        )
    local = torch.cat(chunks[rank::size], dim=dim).contiguous()

    expected_shape = list(full.shape)
    expected_shape[dim] //= size
    if tuple(local.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"inverse partition produced shape {tuple(local.shape)}, "
            f"expected {tuple(expected_shape)}"
        )
    return local


def split_fp32_to_bf16_remainder(
    authoritative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode contiguous FP32 values as TE v2.17 BF16 + INT16 remainder.

    The result is ``(rounded_high_bf16, low_word_int16)``.  This is a bitwise
    transform: finite values, signed zero, infinities, and every NaN payload
    can all be reconstructed exactly.  Inputs are detached, and the input is
    never mutated.
    """

    authoritative = _require_dense_contiguous("authoritative", authoritative)
    if authoritative.dtype != torch.float32:
        raise TypeError(
            f"authoritative must have dtype torch.float32, got {authoritative.dtype}"
        )

    # TE's CUDA union is { float fp32; int16_t int16[2]; }, where int16[0]
    # is the low word and int16[1] the high word.  Integer masks express the
    # same bit layout without relying on a normal BF16 numerical conversion.
    bits = authoritative.detach().view(torch.int32)
    low_u16 = torch.bitwise_and(bits, 0xFFFF)
    high_u16 = torch.bitwise_and(torch.bitwise_right_shift(bits, 16), 0xFFFF)

    # adam.cu: if (local_p_rem[ii] < 0) local_p[ii]++;  // Round up
    round_up = torch.bitwise_and(low_u16, 0x8000).ne(0).to(torch.int32)
    rounded_high_u16 = torch.bitwise_and(high_u16 + round_up, 0xFFFF)

    remainder = low_u16.to(torch.int16).contiguous()
    rounded_high_bits = rounded_high_u16.to(torch.int16).contiguous()
    rounded_high = rounded_high_bits.view(torch.bfloat16)
    return rounded_high, remainder


def reconstruct_fp32_from_bf16_remainder(
    rounded_high: torch.Tensor,
    remainder: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct FP32 values bit-exactly from TE BF16 + INT16 state."""

    rounded_high = _require_dense_contiguous("rounded_high", rounded_high)
    remainder = _require_dense_contiguous("remainder", remainder)
    if rounded_high.dtype != torch.bfloat16:
        raise TypeError(
            f"rounded_high must have dtype torch.bfloat16, got {rounded_high.dtype}"
        )
    if remainder.dtype != torch.int16:
        raise TypeError(f"remainder must have dtype torch.int16, got {remainder.dtype}")
    if rounded_high.shape != remainder.shape:
        raise ValueError(
            f"rounded_high shape {tuple(rounded_high.shape)} does not match "
            f"remainder shape {tuple(remainder.shape)}"
        )
    if rounded_high.device != remainder.device:
        raise ValueError(
            f"rounded_high is on {rounded_high.device}, remainder is on {remainder.device}"
        )

    stored_high_u16 = torch.bitwise_and(
        rounded_high.detach().view(torch.int16).to(torch.int32), 0xFFFF
    )
    low_u16 = torch.bitwise_and(remainder.detach().to(torch.int32), 0xFFFF)

    # adam.cu: if (local_p_rem[ii] < 0) local_p[ii]--;  // Undo rounding
    undo_round = remainder.detach().lt(0).to(torch.int32)
    raw_high_u16 = torch.bitwise_and(stored_high_u16 - undo_round, 0xFFFF)
    raw_bits = torch.bitwise_or(
        torch.bitwise_left_shift(raw_high_u16, 16), low_u16
    ).contiguous()
    return raw_bits.view(torch.float32)


def select_master_param_storage(
    optimizer: object,
    model_param: torch.Tensor,
    *,
    optimizer_state: Mapping[str, object] | None = None,
) -> MasterParamStorage:
    """Select TE's FP32-master or BF16-remainder tensor without mutation.

    ``optimizer`` must be the effective Transformer Engine FusedAdam instance,
    not a Megatron wrapper.  Only public TE v2.17 flags are inspected:
    ``master_weights``, ``master_weight_dtype``, and
    ``store_param_remainders``.  Supplying ``optimizer_state`` is convenient
    for callers that already resolved ``optimizer.state[model_param]``.

    This helper intentionally does not write the model parameter, master
    state, optimizer moments, step counters, or scaling metadata.
    """

    model_param = _require_dense_contiguous("model_param", model_param)
    if model_param.dtype != torch.bfloat16:
        raise ValueError(
            f"Miles value-state installation currently requires a BF16 model "
            f"parameter, got {model_param.dtype}"
        )

    master_weights = getattr(optimizer, "master_weights", None)
    if type(master_weights) is not bool:
        raise ValueError("optimizer.master_weights must be an explicit bool")
    if not master_weights:
        raise ValueError("optimizer does not own master weights")

    master_weight_dtype = getattr(optimizer, "master_weight_dtype", None)
    if master_weight_dtype != torch.float32:
        raise ValueError(
            "only an FP32 TE master is supported; "
            f"optimizer.master_weight_dtype={master_weight_dtype!r}"
        )

    stores_remainders = getattr(optimizer, "store_param_remainders", None)
    if type(stores_remainders) is not bool:
        raise ValueError("optimizer.store_param_remainders must be an explicit bool")

    if optimizer_state is None:
        all_state = getattr(optimizer, "state", None)
        if not isinstance(all_state, Mapping):
            raise ValueError("optimizer.state must be a mapping")
        try:
            optimizer_state = all_state[model_param]
        except KeyError as exc:
            raise ValueError("optimizer has no initialized state for model_param") from exc
    if not isinstance(optimizer_state, Mapping):
        raise TypeError("optimizer_state must be a mapping")
    if "master_param" not in optimizer_state:
        raise ValueError("optimizer_state has no initialized 'master_param' tensor")

    master_param = optimizer_state["master_param"]
    master_param = _require_dense_contiguous("optimizer_state['master_param']", master_param)
    if master_param.shape != model_param.shape:
        raise ValueError(
            f"master_param shape {tuple(master_param.shape)} does not match "
            f"model_param shape {tuple(model_param.shape)}"
        )

    if stores_remainders:
        if master_param.dtype != torch.int16:
            raise ValueError(
                "optimizer flags select BF16 remainders, but master_param has "
                f"dtype {master_param.dtype} instead of torch.int16"
            )
        kind: MasterParamStorageKind = "bf16_remainder"
    else:
        if master_param.dtype != torch.float32:
            raise ValueError(
                "optimizer flags select an FP32 master, but master_param has "
                f"dtype {master_param.dtype}"
            )
        kind = "fp32_master"

    return MasterParamStorage(kind=kind, tensor=master_param)


__all__ = [
    "MasterParamStorage",
    "MasterParamStorageKind",
    "inverse_tp_partition",
    "normalize_miles_partition",
    "reconstruct_fp32_from_bf16_remainder",
    "select_master_param_storage",
    "split_fp32_to_bf16_remainder",
]
