"""Miles/Megatron full-model value-pretrain island adapter for IsoLoCo.

This module is loaded through Miles's model/step hooks.  Miles remains the
training stack (value targets, loss masks, 256K packing, optimizer, scheduler,
checkpointing, and validation); Yeto only synchronizes complete canonical
parameter tensors between independent TP4 x CP2, DP1 islands.

The implementation deliberately imports Miles and Megatron lazily so its
layout/config helpers remain CPU-testable.  Every model rank enters every
island boundary.  Only logical CP-rank 0 / TP-rank 0 owns the WAN client.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable

import torch
import torch.distributed as dist

from ..fragments import (
    FRAGMENT_PATTERNS,
    MERGE_AVG,
    MERGE_ISO,
    Fragment,
    FragmentLayout,
    build_layout,
    is_embedding_name,
)
from ..protocol import (
    DTYPE_F32,
    DTYPE_Q4,
    SyncerClient,
    bulk_dtype,
    layout_fingerprint,
)
from ..tensor_io import pack_tensor, quantize_q4, unpack_fragment
from .miles_value_state import (
    inverse_tp_partition,
    normalize_miles_partition,
    reconstruct_fp32_from_bf16_remainder,
    split_fp32_to_bf16_remainder,
)


log = logging.getLogger("yeto.miles-value-island")


_GRAD_DTYPES = (torch.bfloat16, torch.float32)


def _grouped_grad_norm(
    original_get_grad_norm: Callable[..., float],
    grads_for_norm: list[torch.Tensor] | torch.Tensor,
    norm_type: int | float = 2,
    grad_stats_parallel_group: dist.ProcessGroup | None = None,
) -> float:
    """Run MCore's multi-tensor norm once per homogeneous gradient dtype.

    Transformer Engine chooses one CUDA template from the first tensor in a
    multi-tensor list. A precision-aware optimizer can expose both BF16 and
    FP32 ``decoupled_grad`` tensors, so sending the mixed list directly makes
    the kernel reinterpret later tensor addresses with the wrong dtype.

    Every rank calls both dtype groups in the same fixed order, including an
    empty group. MCore's norm helper performs a collective even for an empty
    local list, so conditionally skipping a group could deadlock ranks whose
    TP-filtered parameter sets have different dtype composition.
    """

    if isinstance(grads_for_norm, torch.Tensor):
        grads = [grads_for_norm]
    else:
        grads = list(grads_for_norm)
    grouped = {dtype: [] for dtype in _GRAD_DTYPES}
    for grad in grads:
        if grad.dtype not in grouped:
            raise TypeError(
                "precision-aware grad norm supports only BF16/FP32 grads; got "
                f"{grad.dtype}"
            )
        grouped[grad.dtype].append(grad)

    norm_type_float = float(norm_type)
    group_norms = [
        float(
            original_get_grad_norm(
                grouped[dtype],
                norm_type=norm_type,
                grad_stats_parallel_group=grad_stats_parallel_group,
            )
        )
        for dtype in _GRAD_DTYPES
    ]
    if norm_type_float == math.inf:
        return max(group_norms)
    if norm_type_float == 2.0:
        return math.hypot(*group_norms)
    return sum(value**norm_type_float for value in group_norms) ** (
        1.0 / norm_type_float
    )


def _grouped_clip_grad(
    original_clip_grad: Callable[..., None],
    parameters: list[torch.Tensor] | torch.Tensor,
    max_norm: int | float,
    total_norm: float,
    use_decoupled_grad: bool = False,
) -> None:
    """Run MCore's multi-tensor scale on homogeneous decoupled-grad lists."""

    if not use_decoupled_grad:
        original_clip_grad(parameters, max_norm, total_norm, False)
        return
    if isinstance(parameters, torch.Tensor):
        params = [parameters]
    else:
        params = list(parameters)
    grouped = {dtype: [] for dtype in _GRAD_DTYPES}
    for parameter in params:
        grad = getattr(parameter, "decoupled_grad", None)
        if grad is None:
            continue
        if grad.dtype not in grouped:
            raise TypeError(
                "precision-aware grad clipping supports only BF16/FP32 grads; got "
                f"{grad.dtype}"
            )
        grouped[grad.dtype].append(parameter)
    for dtype in _GRAD_DTYPES:
        if grouped[dtype]:
            original_clip_grad(
                grouped[dtype],
                max_norm,
                total_norm,
                use_decoupled_grad=True,
            )


def _install_mixed_dtype_grad_compat(args: Any) -> None:
    """Patch MCore's imported norm/clip aliases for BF16 grad reduction."""

    if not (
        getattr(args, "bf16", False)
        and getattr(args, "grad_reduce_in_bf16", False)
        and getattr(args, "use_precision_aware_optimizer", False)
    ):
        return

    from megatron.core.optimizer import clip_grads as clip_grads_module
    from megatron.core.optimizer import optimizer as optimizer_module

    if getattr(optimizer_module, "_yeto_mixed_dtype_grad_compat", False):
        return
    original_get_grad_norm = optimizer_module.get_grad_norm_fp32
    original_clip_grad = optimizer_module.clip_grad_by_total_norm_fp32

    def get_grad_norm_compat(
        grads_for_norm: list[torch.Tensor] | torch.Tensor,
        norm_type: int | float = 2,
        grad_stats_parallel_group: dist.ProcessGroup | None = None,
    ) -> float:
        return _grouped_grad_norm(
            original_get_grad_norm,
            grads_for_norm,
            norm_type,
            grad_stats_parallel_group,
        )

    def clip_grad_compat(
        parameters: list[torch.Tensor] | torch.Tensor,
        max_norm: int | float,
        total_norm: float,
        use_decoupled_grad: bool = False,
    ) -> None:
        _grouped_clip_grad(
            original_clip_grad,
            parameters,
            max_norm,
            total_norm,
            use_decoupled_grad,
        )

    # optimizer.py imports these names directly, while other callers may use
    # clip_grads.py. Keep both module aliases consistent within this process.
    optimizer_module.get_grad_norm_fp32 = get_grad_norm_compat
    optimizer_module.clip_grad_by_total_norm_fp32 = clip_grad_compat
    clip_grads_module.get_grad_norm_fp32 = get_grad_norm_compat
    clip_grads_module.clip_grad_by_total_norm_fp32 = clip_grad_compat
    optimizer_module._yeto_mixed_dtype_grad_compat = True
    log.info("installed homogeneous BF16/FP32 grad norm and clipping compatibility")


def _install_bf16_static_unscale_compat(args: Any, optimizer: Any) -> None:
    """Use PyTorch AMP unscale semantics for BF16 precision-aware grads.

    PyTorch's CUDA ``_amp_foreach_non_finite_check_and_unscale_`` still does
    not dispatch BF16, while Megatron's precision-aware distributed optimizer
    exposes BF16 ``decoupled_grad`` tensors when gradient reduction is BF16.
    Keep Megatron's configured static loss scale unchanged and provide the
    missing BF16 dispatch in Python. FP32 gradients continue through the
    native fused op; BF16 gradients are checked and unscaled in place, so this
    does not materialize a full FP32 gradient buffer.
    """

    if not (
        getattr(args, "bf16", False)
        and getattr(args, "grad_reduce_in_bf16", False)
        and getattr(args, "loss_scale", None) is not None
    ):
        return

    def unscale_and_check(instance: Any) -> bool:
        main_grads = (
            []
            if instance.is_stub_optimizer
            else instance._collect_main_grad_data_for_unscaling()
        )
        instance.found_inf.fill_(0.0)

        fp32_grads = [grad for grad in main_grads if grad.dtype == torch.float32]
        bf16_grads = [grad for grad in main_grads if grad.dtype == torch.bfloat16]
        unsupported = [
            grad.dtype
            for grad in main_grads
            if grad.dtype not in (torch.float32, torch.bfloat16)
        ]
        if unsupported:
            raise TypeError(
                "static unscale supports only FP32/BF16 main grads; got "
                f"{sorted({str(dtype) for dtype in unsupported})}"
            )

        if fp32_grads:
            torch._amp_foreach_non_finite_check_and_unscale_(
                fp32_grads, instance.found_inf, instance.grad_scaler.inv_scale
            )
            # Native AMP checks each FP32 input before multiplying by
            # inv_scale, but a static loss scale below one makes that
            # multiplication enlarge gradients. Catch the (theoretical but
            # otherwise silent) case where a finite input overflows here.
            fp32_finite_after = torch.stack(
                [torch.isfinite(grad).all() for grad in fp32_grads]
            ).all()
            fp32_found_inf = (
                (~fp32_finite_after)
                .to(dtype=instance.found_inf.dtype)
                .reshape_as(instance.found_inf)
            )
            instance.found_inf.copy_(torch.maximum(instance.found_inf, fp32_found_inf))
        if bf16_grads:
            # Native AMP checks the scaled input before unscale. Our scale is
            # below one, so inv_scale enlarges gradients; check both sides to
            # reject a finite scaled value that becomes non-finite while
            # unscaling. Keep all reductions on device and avoid one host sync
            # per parameter.
            finite_before = torch.stack(
                [torch.isfinite(grad).all() for grad in bf16_grads]
            ).all()
            torch._foreach_mul_(bf16_grads, instance.grad_scaler.inv_scale.reshape(()))
            finite_after = torch.stack(
                [torch.isfinite(grad).all() for grad in bf16_grads]
            ).all()
            bf16_found_inf = (
                (~(finite_before & finite_after))
                .to(dtype=instance.found_inf.dtype)
                .reshape_as(instance.found_inf)
            )
            instance.found_inf.copy_(torch.maximum(instance.found_inf, bf16_found_inf))

        dist.all_reduce(
            instance.found_inf,
            op=dist.ReduceOp.MAX,
            group=instance.get_grad_stats_parallel_group(),
        )
        return instance.found_inf.item() > 0

    installed = 0
    for item in optimizer.chained_optimizers:
        config = getattr(item, "config", None)
        if not getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False):
            continue
        if getattr(item, "grad_scaler", None) is None:
            raise RuntimeError(
                "BF16 static-unscale compatibility requires a configured grad scaler"
            )
        item._unscale_main_grads_and_check_for_nan = MethodType(unscale_and_check, item)
        installed += 1
    if installed == 0:
        raise RuntimeError(
            "BF16 static-unscale compatibility found no precision-aware optimizer"
        )
    log.info(
        "installed BF16 static-unscale compatibility on %d optimizer(s)", installed
    )


@dataclass(frozen=True)
class TensorDescriptor:
    """Canonical identity and TP partition metadata for one model parameter."""

    name: str
    local_shape: tuple[int, ...]
    full_shape: tuple[int, ...]
    tp_sharded: bool
    partition_dim: int | None
    partition_stride: int
    merge_mode: int

    @property
    def numel(self) -> int:
        value = 1
        for dim in self.full_shape:
            value *= dim
        return value


@dataclass
class _RuntimeTensor:
    descriptor: TensorDescriptor
    model_param: torch.nn.Parameter
    dist_optimizer: Any
    adam_optimizer: Any
    optimizer_param: torch.Tensor | None
    owned_start: int | None
    owned_end: int | None
    ownership_ranges: tuple[tuple[int, int] | None, ...]


@dataclass(frozen=True)
class _FragmentTensorSpan:
    """One complete canonical tensor's exact interval in a flat fragment."""

    name: str
    start: int
    end: int
    full_shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return self.end - self.start


def _validate_owned_ranges(
    name: str,
    numel: int,
    ranges: list[tuple[int, int] | None],
) -> None:
    """Require DP/CP optimizer shards to partition one TP-local parameter."""

    owned = sorted(item for item in ranges if item is not None)
    if not owned:
        raise RuntimeError(f"{name}: no optimizer rank owns any parameter values")
    cursor = 0
    for start, end in owned:
        if not (0 <= start < end <= numel):
            raise RuntimeError(
                f"{name}: invalid optimizer ownership range [{start}, {end}) "
                f"for {numel} values"
            )
        if start != cursor:
            relation = "overlap" if start < cursor else "gap"
            raise RuntimeError(
                f"{name}: optimizer ownership {relation} at value {cursor}; "
                f"next range is [{start}, {end})"
            )
        cursor = end
    if cursor != numel:
        raise RuntimeError(
            f"{name}: optimizer ownership ends at {cursor}, expected {numel}"
        )


def grouped_tensor_fragment_layout(
    descriptors: list[TensorDescriptor],
    num_fragments: int = 96,
    pattern: str = "binpack",
) -> FragmentLayout:
    """Group complete canonical tensors into a deterministic Iso layout.

    No tensor is split. Two-dimensional non-embedding tensors use Iso;
    vectors and embedding-like tensors use direct averaging. A 1xH value
    head is still an Iso tensor, for which spectrum flattening is
    mathematically identity. ``build_layout`` gives both supported patterns
    deterministic name-based tie breaking and keeps merge modes separate.
    """

    if not descriptors:
        raise ValueError("Miles value island has no trainable parameters")
    if num_fragments < 1:
        raise ValueError("num_fragments must be >= 1")
    if pattern not in FRAGMENT_PATTERNS:
        raise ValueError(
            f"pattern must be one of {FRAGMENT_PATTERNS}, got {pattern!r}"
        )
    ordered = sorted(descriptors, key=lambda item: item.name)
    if len({item.name for item in ordered}) != len(ordered):
        raise ValueError("Miles value island parameter names are not unique")
    for item in ordered:
        expected = 1
        for dim in item.full_shape:
            if dim <= 0:
                raise ValueError(f"{item.name}: canonical shape must be positive")
            expected *= dim
        if expected != item.numel:
            raise RuntimeError(f"{item.name}: inconsistent canonical numel")
        if item.merge_mode == MERGE_ISO:
            if len(item.full_shape) != 2:
                raise ValueError(f"{item.name}: Iso requires a two-dimensional tensor")
        elif item.merge_mode != MERGE_AVG:
            raise ValueError(f"{item.name}: unsupported merge mode {item.merge_mode}")

    layout = build_layout(
        [(item.name, item.numel) for item in ordered],
        num_fragments,
        pattern,
        matrix_merge="iso",
        named_shapes={item.name: item.full_shape for item in ordered},
    )
    by_name = {item.name: item for item in ordered}
    for fragment in layout.fragments:
        for name, numel in fragment.tensors:
            item = by_name[name]
            if numel != item.numel:
                raise RuntimeError(
                    f"{name}: layout has {numel} values, expected {item.numel}"
                )
            if fragment.merge_mode != item.merge_mode:
                raise RuntimeError(
                    f"{name}: layout merge mode {fragment.merge_mode} differs "
                    f"from descriptor mode {item.merge_mode}"
                )
    if sorted(layout.tensor_names()) != [item.name for item in ordered]:
        raise RuntimeError("grouped fragment layout lost or duplicated parameters")
    return layout


def one_tensor_fragment_layout(
    descriptors: list[TensorDescriptor],
) -> FragmentLayout:
    """Compatibility layout with exactly one complete tensor per fragment."""

    grouped_tensor_fragment_layout(
        descriptors,
        num_fragments=len(descriptors),
        pattern="binpack",
    )
    fragments = []
    for item in sorted(descriptors, key=lambda descriptor: descriptor.name):
        fragments.append(
            Fragment(
                merge_mode=item.merge_mode,
                tensors=[(item.name, item.numel)],
                shapes=(
                    {item.name: (item.full_shape[0], item.full_shape[1])}
                    if item.merge_mode == MERGE_ISO
                    else None
                ),
                identity_shapes={item.name: item.full_shape},
            )
        )
    return FragmentLayout(fragments)


def _fragment_tensor_spans(
    fragment: Fragment,
    descriptors_by_name: dict[str, TensorDescriptor],
) -> tuple[_FragmentTensorSpan, ...]:
    """Resolve and validate deterministic flat offsets for one fragment."""

    spans: list[_FragmentTensorSpan] = []
    seen: set[str] = set()
    offset = 0
    for name, declared_numel in fragment.tensors:
        if name in seen:
            raise ValueError(f"fragment contains duplicate tensor {name!r}")
        seen.add(name)
        try:
            descriptor = descriptors_by_name[name]
        except KeyError as exc:
            raise ValueError(f"fragment references unknown tensor {name!r}") from exc
        if declared_numel != descriptor.numel:
            raise ValueError(
                f"{name}: fragment declares {declared_numel} values, "
                f"descriptor has {descriptor.numel}"
            )
        identity_shape = (
            None
            if fragment.identity_shapes is None
            else fragment.identity_shapes.get(name)
        )
        if identity_shape != descriptor.full_shape:
            raise ValueError(
                f"{name}: fragment identity shape {identity_shape} differs from "
                f"canonical shape {descriptor.full_shape}"
            )
        if fragment.merge_mode != descriptor.merge_mode:
            raise ValueError(
                f"{name}: fragment merge mode {fragment.merge_mode} differs from "
                f"descriptor mode {descriptor.merge_mode}"
            )
        if descriptor.merge_mode == MERGE_ISO:
            shape = None if fragment.shapes is None else fragment.shapes.get(name)
            if shape != descriptor.full_shape:
                raise ValueError(
                    f"{name}: Iso shape {shape} differs from canonical shape "
                    f"{descriptor.full_shape}"
                )
        spans.append(
            _FragmentTensorSpan(
                name=name,
                start=offset,
                end=offset + declared_numel,
                full_shape=descriptor.full_shape,
            )
        )
        offset += declared_numel
    if offset != fragment.numel:
        raise RuntimeError(
            f"fragment spans cover {offset} values, expected {fragment.numel}"
        )
    return tuple(spans)


def _validate_flat_fragment(
    fragment: Fragment,
    flat: torch.Tensor,
    *,
    context: str,
) -> torch.Tensor:
    """Require an exact one-dimensional payload for the whole fragment."""

    if flat.ndim != 1 or flat.numel() != fragment.numel:
        raise ValueError(
            f"{context}: flat fragment has shape {tuple(flat.shape)}, "
            f"expected ({fragment.numel},)"
        )
    return flat


def _pack_flat_fragment(fragment: Fragment, flat: torch.Tensor, dtype: int) -> bytes:
    return pack_tensor(
        _validate_flat_fragment(fragment, flat, context="pack"),
        dtype,
    )


def _unpack_flat_fragment(fragment: Fragment, data: bytes, dtype: int) -> torch.Tensor:
    return _validate_flat_fragment(
        fragment,
        unpack_fragment(fragment, data, dtype),
        context="unpack",
    )


def _quantize_flat_fragment(fragment: Fragment, flat: torch.Tensor) -> bytes:
    return quantize_q4(
        _validate_flat_fragment(fragment, flat, context="quantize")
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} must be set for the Miles value island")
    return value.strip()


def _env_int(name: str, default: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        if default is None:
            raise ValueError(f"{name} must be set")
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    return value


def _required_positive_env_int(name: str) -> int:
    value = _env_int(name)
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _parse_syncer(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator or not host:
        raise ValueError(f"YETO_VALUE_SYNCER must be host:port, got {value!r}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"bad YETO_VALUE_SYNCER port in {value!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"YETO_VALUE_SYNCER port must be in [1, 65535], got {port}")
    return host, port


class MilesValueIsland:
    """One TP4 x CP2 Miles learner and its single WAN Yeto client."""

    def __init__(self, args: Any, model: Any, optimizer: Any):
        if optimizer is None:
            raise ValueError("Miles value IsoLoCo requires a real optimizer")
        if not dist.is_initialized():
            raise RuntimeError(
                "Miles value IsoLoCo requires initialized torch.distributed"
            )

        # Runtime-only imports: keep module importable in CPU unit tests.
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
        from miles.backends.megatron_utils.update_weight.common import (
            _gather_with_stride,
            named_params_and_buffers,
        )
        from miles.backends.training_utils.parallel import get_parallel_state

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self._gather_with_stride: Callable[..., torch.Tensor] = _gather_with_stride
        self.parallel = get_parallel_state()
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device("cuda", torch.cuda.current_device())

        self._validate_topology()
        self.is_leader = self.parallel.cp.rank == 0 and self.parallel.tp.rank == 0
        leader_marker = torch.tensor(
            [self.global_rank + 1 if self.is_leader else 0],
            dtype=torch.int64,
            device=self.device,
        )
        dist.all_reduce(leader_marker, op=dist.ReduceOp.MAX)
        if int(leader_marker.item()) == 0:
            raise RuntimeError("Miles value island has no CP0/TP0 leader")
        self.leader_global_rank = int(leader_marker.item()) - 1

        self.tp_source_global = self._logical_group_source(
            self.parallel.tp.group, self.parallel.tp.rank, 0, "TP"
        )
        self.cp_source_global = self._logical_group_source(
            self.parallel.cp.group, self.parallel.cp.rank, 0, "CP"
        )

        dist_optimizers = [
            item
            for item in optimizer.chained_optimizers
            if isinstance(item, DistributedOptimizer)
        ]
        if len(dist_optimizers) != len(optimizer.chained_optimizers):
            raise TypeError("every Miles value optimizer must be DistributedOptimizer")

        named = list(
            named_params_and_buffers(
                args,
                model,
                convert_to_global_name=True,
                translate_gpu_to_cpu=False,
            )
        )
        runtimes: list[_RuntimeTensor] = []
        seen_names: set[str] = set()
        for name, parameter in named:
            if (
                not isinstance(parameter, torch.nn.Parameter)
                or not parameter.requires_grad
            ):
                continue
            if name in seen_names:
                raise ValueError(f"duplicate Miles global parameter name {name!r}")
            seen_names.add(name)
            descriptor = self._describe_parameter(name, parameter)

            # ``model_param_group_index_map`` contains only parameters with a
            # locally-owned ZeRO slice.  Under CP2 a parameter can therefore
            # be absent on one rank even though this DistributedOptimizer is
            # its owner.  The underlying model buffers retain the complete
            # parameter catalog on every member of the DP-with-CP group.
            owners = [
                item
                for item in dist_optimizers
                if any(parameter in buffer.param_index_map for buffer in item.buffers)
            ]
            if len(owners) != 1:
                raise RuntimeError(
                    f"{name}: expected exactly one distributed optimizer owner, got {len(owners)}"
                )
            dist_optimizer = owners[0]
            if not dist_optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                raise RuntimeError(f"{name}: precision-aware optimizer is not active")
            adam = dist_optimizer.optimizer
            if parameter.dtype not in (torch.bfloat16, torch.float32):
                raise TypeError(f"{name}: unsupported model dtype {parameter.dtype}")
            if not getattr(adam, "master_weights", False):
                raise RuntimeError(f"{name}: TE FusedAdam master_weights is disabled")
            if getattr(adam, "master_weight_dtype", None) != torch.float32:
                raise RuntimeError(f"{name}: TE master_weight_dtype is not FP32")
            if parameter.dtype == torch.bfloat16 and not getattr(
                adam, "store_param_remainders", False
            ):
                raise RuntimeError(f"{name}: TE BF16 remainder storage is disabled")

            optimizer_param: torch.Tensor | None = None
            owned_start: int | None = None
            owned_end: int | None = None
            if parameter in dist_optimizer.model_param_gbuf_map:
                range_map = dist_optimizer._get_model_param_range_map(parameter)
                owned_start = int(range_map["param"].start)
                owned_end = int(range_map["param"].end)
                group_index, group_order = dist_optimizer.model_param_group_index_map[
                    parameter
                ]
                optimizer_param = adam.param_groups[group_index]["params"][group_order]
                owned_numel = owned_end - owned_start
                if optimizer_param.numel() != owned_numel:
                    raise RuntimeError(
                        f"{name}: optimizer shard has {optimizer_param.numel()} values, "
                        f"but ownership range [{owned_start}, {owned_end}) has "
                        f"{owned_numel}"
                    )
                if optimizer_param.dtype != parameter.dtype:
                    raise TypeError(
                        f"{name}: optimizer/model dtype mismatch "
                        f"{optimizer_param.dtype} != {parameter.dtype}"
                    )
            elif parameter in dist_optimizer.model_param_group_index_map:
                raise RuntimeError(
                    f"{name}: optimizer group entry exists without a local gbuf range"
                )

            local_range = (
                None
                if owned_start is None or owned_end is None
                else (owned_start, owned_end)
            )
            ownership: list[tuple[int, int] | None] = [None] * dist.get_world_size(
                group=dist_optimizer.data_parallel_group
            )
            dist.all_gather_object(
                ownership,
                local_range,
                group=dist_optimizer.data_parallel_group,
            )
            _validate_owned_ranges(name, parameter.numel(), ownership)
            runtimes.append(
                _RuntimeTensor(
                    descriptor=descriptor,
                    model_param=parameter,
                    dist_optimizer=dist_optimizer,
                    adam_optimizer=adam,
                    optimizer_param=optimizer_param,
                    owned_start=owned_start,
                    owned_end=owned_end,
                    ownership_ranges=tuple(ownership),
                )
            )

        runtimes.sort(key=lambda item: item.descriptor.name)
        self.tensors = runtimes
        self.descriptors = [item.descriptor for item in runtimes]
        self.tensors_by_name = {
            item.descriptor.name: item for item in self.tensors
        }
        self.descriptors_by_name = {
            item.name: item for item in self.descriptors
        }
        self._validate_catalog_across_ranks()
        requested_fragments = _env_int("YETO_VALUE_NUM_FRAGMENTS", 96)
        fragment_pattern = os.environ.get(
            "YETO_VALUE_FRAGMENT_PATTERN", "binpack"
        ).strip()
        self.layout = grouped_tensor_fragment_layout(
            self.descriptors,
            num_fragments=requested_fragments,
            pattern=fragment_pattern,
        )
        self.fragment_spans = [
            _fragment_tensor_spans(fragment, self.descriptors_by_name)
            for fragment in self.layout.fragments
        ]
        flattened_names = [
            span.name for spans in self.fragment_spans for span in spans
        ]
        if sorted(flattened_names) != sorted(self.tensors_by_name):
            raise RuntimeError("grouped fragment spans lost or duplicated parameters")
        self.layout_fingerprint = self._validate_layout_across_ranks()

        self.learner_id = _env_int("YETO_VALUE_LEARNER_ID")
        self.num_learners = _env_int("YETO_VALUE_NUM_LEARNERS", 5)
        if not 0 <= self.learner_id < self.num_learners:
            raise ValueError(
                f"YETO_VALUE_LEARNER_ID={self.learner_id} is outside "
                f"[0, {self.num_learners})"
            )
        self.merge_alpha = _env_float("YETO_VALUE_MERGE_ALPHA", 0.5)
        if not 0.0 <= self.merge_alpha <= 1.0:
            raise ValueError("YETO_VALUE_MERGE_ALPHA must be in [0, 1]")
        self.budget_steps = (
            _env_int("YETO_VALUE_BUDGET_STEPS")
            if os.environ.get("YETO_VALUE_BUDGET_STEPS")
            else None
        )
        if self.budget_steps is not None and self.budget_steps <= 0:
            raise ValueError("YETO_VALUE_BUDGET_STEPS must be positive")
        # The syncer can pipeline pull requests before it has measured the
        # learner step rate.  Enforce H again at the learner, per fragment and
        # relative to that fragment's last installed global anchor.  Requiring
        # this explicit contract fails closed instead of silently reverting to
        # the historical one-step readiness threshold.
        self.min_local_steps = _required_positive_env_int(
            "YETO_VALUE_MIN_LOCAL_STEPS"
        )
        self.steps_total = _env_int("YETO_VALUE_LOCAL_STEP_OFFSET", 0)
        self.units_total = _env_int("YETO_VALUE_UNIT_OFFSET", 0)
        if self.steps_total < 0 or self.units_total < 0:
            raise ValueError("Yeto resume offsets cannot be negative")

        count = self.layout.num_fragments
        self.steps_at_reset = [self.steps_total] * count
        self.units_at_reset = [self.units_total] * count
        self.fragment_versions = [0] * count
        self.anchors: list[torch.Tensor | None] = [None] * count
        self.pending_pulls: list[Any] = []
        self._leader_payloads: dict[tuple[int, int], bytes] = {}
        self._fresh_state_dist_optimizers: dict[int, Any] = {}
        self.initial_ready = False
        self.finalized = False

        def start_client() -> SyncerClient:
            client = SyncerClient(
                _parse_syncer(_required_env("YETO_VALUE_SYNCER")),
                self.learner_id,
                self.layout,
                dtype=DTYPE_Q4,
                num_streams=_env_int("YETO_VALUE_STREAMS", 4),
                connect_timeout=_env_float("YETO_VALUE_CONNECT_TIMEOUT", 3600.0),
                max_reconnects=None,
                finalization_timeout=_env_float(
                    "YETO_VALUE_FINALIZATION_TIMEOUT", 3600.0
                ),
            )
            client.start()
            return client

        self.client: SyncerClient | None = self._leader_local(start_client)

        self._send_initial_parameters()
        if self.is_leader:
            log.info(
                "initialized Miles value island learner=%d tensors=%d fragments=%d "
                "pattern=%s TP=%d CP=%d",
                self.learner_id,
                len(self.tensors),
                self.layout.num_fragments,
                fragment_pattern,
                self.parallel.tp.size,
                self.parallel.cp.size,
            )

    def _validate_topology(self) -> None:
        checks = {
            "TP": (self.parallel.tp.size, 4),
            "CP": (self.parallel.cp.size, 2),
            "PP": (self.parallel.pp.size, 1),
            "DP": (self.parallel.intra_dp.size, 1),
            "EP": (self.parallel.ep.size, 1),
            "ETP": (self.parallel.etp.size, 1),
        }
        wrong = [
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in checks.items()
            if actual != expected
        ]
        if wrong:
            raise RuntimeError(
                "Miles value island topology mismatch: " + ", ".join(wrong)
            )
        if self.world_size != 8:
            raise RuntimeError(
                f"Miles value island world size must be 8, got {self.world_size}"
            )

    def _logical_group_source(
        self, group: Any, logical_rank: int, wanted: int, label: str
    ) -> int:
        entries: list[Any] = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(entries, (logical_rank, self.global_rank), group=group)
        matches = [global_rank for rank, global_rank in entries if rank == wanted]
        if len(matches) != 1:
            raise RuntimeError(
                f"{label} group has {len(matches)} logical rank-{wanted} members"
            )
        return int(matches[0])

    def _describe_parameter(
        self, name: str, parameter: torch.nn.Parameter
    ) -> TensorDescriptor:
        if not parameter.is_contiguous() or parameter.ndim == 0:
            raise ValueError(f"{name}: parameter must be non-scalar and contiguous")
        local_shape = tuple(int(dim) for dim in parameter.shape)
        tp_sharded = (
            bool(getattr(parameter, "tensor_model_parallel", False))
            and getattr(parameter, "parallel_mode", None) != "duplicated"
        )
        if tp_sharded:
            stride, dim = normalize_miles_partition(
                name,
                swiglu=bool(self.args.swiglu),
                partition_stride=int(getattr(parameter, "partition_stride")),
                partition_dim=int(getattr(parameter, "partition_dim")),
            )
            if dim < -parameter.ndim or dim >= parameter.ndim:
                raise ValueError(f"{name}: invalid partition_dim {dim}")
            dim %= parameter.ndim
            full_shape = list(local_shape)
            full_shape[dim] *= self.parallel.tp.size
            partition_dim: int | None = dim
        else:
            stride = 1
            partition_dim = None
            full_shape = list(local_shape)
        full_shape_tuple = tuple(full_shape)
        merge_mode = (
            MERGE_ISO
            if len(full_shape_tuple) == 2 and not is_embedding_name(name)
            else MERGE_AVG
        )
        return TensorDescriptor(
            name=name,
            local_shape=local_shape,
            full_shape=full_shape_tuple,
            tp_sharded=tp_sharded,
            partition_dim=partition_dim,
            partition_stride=stride,
            merge_mode=merge_mode,
        )

    def _validate_catalog_across_ranks(self) -> None:
        catalogs: list[Any] = [None] * self.world_size
        dist.all_gather_object(catalogs, self.descriptors)
        reference = catalogs[0]
        for rank, catalog in enumerate(catalogs[1:], start=1):
            if catalog != reference:
                raise RuntimeError(
                    f"Miles value parameter catalog differs on global rank {rank}"
                )

    def _validate_layout_across_ranks(self) -> bytes:
        """Require one authoritative grouped layout on every model rank.

        The digest is the same semantic fingerprint sent in the syncer HELLO:
        it covers fragment order, merge modes, ordered tensor names, lengths,
        and canonical shapes.  Use one fixed 32-byte tensor from every rank so
        different rank-local fragment counts or tensor orders cannot alter the
        collective shape.  Every rank sees the complete result and therefore
        raises before any fragment gather/apply or client startup.
        """

        fingerprint = layout_fingerprint(self.layout)
        if len(fingerprint) != 32:
            raise RuntimeError(
                "Miles value layout fingerprint must contain exactly 32 bytes"
            )
        local = torch.tensor(
            list(fingerprint), dtype=torch.uint8, device=self.device
        )
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local)
        fingerprints = [bytes(item.cpu().tolist()) for item in gathered]
        reference = fingerprints[0]
        mismatched = [
            rank
            for rank, candidate in enumerate(fingerprints)
            if candidate != reference
        ]
        if mismatched:
            observed = ", ".join(
                f"rank {rank}={candidate.hex()}"
                for rank, candidate in enumerate(fingerprints)
            )
            raise RuntimeError(
                "Miles value grouped layout fingerprint mismatch across model "
                f"ranks (different ranks: {mismatched}): {observed}"
            )
        return fingerprint

    def _leader_value(self, builder: Callable[[], Any]) -> Any:
        box: list[Any] = [None]
        if self.is_leader:
            try:
                box[0] = (True, builder())
            except BaseException as exc:
                box[0] = (False, f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(box, src=self.leader_global_rank)
        ok, value = box[0]
        if not ok:
            raise RuntimeError(f"Miles value island leader failed: {value}")
        return value

    def _leader_local(self, builder: Callable[[], Any]) -> Any:
        """Run a leader-only action and propagate its status to every rank.

        Unlike :meth:`_leader_value`, the result remains leader-local.  This
        is used for tensors, clients, and WAN sends that must not be pickled,
        while still ensuring a Python exception cannot make the other seven
        ranks enter the next collective and hang forever.  The success path
        broadcasts one fixed byte; only the exceptional path broadcasts the
        diagnostic string as a Python object.
        """

        result = None
        error = None
        if self.is_leader:
            try:
                result = builder()
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
        status = torch.tensor(
            [1 if self.is_leader and error is None else 0],
            dtype=torch.uint8,
            device=self.device,
        )
        dist.broadcast(status, src=self.leader_global_rank)
        if int(status.item()) == 0:
            diagnostic = [error if self.is_leader else None]
            dist.broadcast_object_list(diagnostic, src=self.leader_global_rank)
            raise RuntimeError(
                f"Miles value island leader failed: {diagnostic[0]}"
            )
        return result

    def _assert_optimizer_state_resident(self, runtime: _RuntimeTensor) -> None:
        offloader = getattr(runtime.dist_optimizer, "_state_offloader", None)
        if offloader is not None and getattr(offloader, "_offloaded", False):
            raise RuntimeError(
                f"{runtime.descriptor.name}: optimizer state is offloaded at Yeto boundary"
            )

    def _local_master(self, runtime: _RuntimeTensor) -> torch.Tensor:
        self._assert_optimizer_state_resident(runtime)
        # DistributedOptimizer shards its contiguous parameter buffers over
        # the DP-with-CP group, independently of model-parameter boundaries.
        # Reconstruct one complete TP-local FP32 master at a time.  Every
        # value has exactly one owner (validated during construction).  Use
        # transport rather than arithmetic SUM so signed zero, subnormals,
        # NaN payloads, and every other FP32 bit pattern remain exact.  This
        # exchange remains entirely within this H200 node.
        full_precision = torch.zeros(
            runtime.model_param.numel(), dtype=torch.float32, device=self.device
        )
        parameter = runtime.optimizer_param
        if parameter is not None:
            assert runtime.owned_start is not None and runtime.owned_end is not None
            state = runtime.adam_optimizer.state.get(parameter)
            if not state or "master_param" not in state:
                # Fresh optimizer before FusedAdam's lazy state initialization.
                owned = parameter.detach().float()
            else:
                master = runtime.adam_optimizer.get_unscaled_state(
                    parameter, "master_param"
                )
                if parameter.dtype == torch.bfloat16:
                    if master.dtype != torch.int16:
                        raise TypeError(
                            f"{runtime.descriptor.name}: expected TE int16 "
                            f"remainder, got {master.dtype}"
                        )
                    owned = reconstruct_fp32_from_bf16_remainder(parameter, master)
                else:
                    if master.dtype != torch.float32:
                        raise TypeError(
                            f"{runtime.descriptor.name}: expected FP32 master, "
                            f"got {master.dtype}"
                        )
                    owned = master.detach()
            if owned.numel() != runtime.owned_end - runtime.owned_start:
                raise RuntimeError(
                    f"{runtime.descriptor.name}: reconstructed optimizer slice has "
                    f"{owned.numel()} values, expected "
                    f"{runtime.owned_end - runtime.owned_start}"
                )
            full_precision[runtime.owned_start : runtime.owned_end].copy_(
                owned.view(-1)
            )
        gathered = [
            torch.empty_like(full_precision)
            for _ in range(len(runtime.ownership_ranges))
        ]
        dist.all_gather(
            gathered,
            full_precision,
            group=runtime.dist_optimizer.data_parallel_group,
        )
        for source, owned_range in zip(gathered, runtime.ownership_ranges):
            if owned_range is None:
                continue
            start, end = owned_range
            full_precision[start:end].copy_(source[start:end])
        return full_precision.view(runtime.descriptor.local_shape).contiguous()

    def _ensure_adam_state(self, runtime: _RuntimeTensor) -> None:
        parameter = runtime.optimizer_param
        if parameter is None:
            raise RuntimeError(
                f"{runtime.descriptor.name}: rank with no optimizer slice cannot "
                "initialize Adam state"
            )
        state = runtime.adam_optimizer.state.get(parameter)
        if state is None or len(state) == 0:
            runtime.adam_optimizer.state[parameter] = {}
            runtime.adam_optimizer.initialize_state(
                parameter,
                bool(
                    parameter.dtype == torch.bfloat16
                    and runtime.adam_optimizer.store_param_remainders
                ),
            )
            fresh = getattr(self, "_fresh_state_dist_optimizers", None)
            if fresh is None:
                fresh = {}
                self._fresh_state_dist_optimizers = fresh
            fresh[id(runtime.dist_optimizer)] = runtime.dist_optimizer
            state = runtime.adam_optimizer.state[parameter]
        required = {"master_param", "exp_avg", "exp_avg_sq"}
        missing = required.difference(state)
        if missing:
            raise RuntimeError(
                f"{runtime.descriptor.name}: initialized TE state is missing {sorted(missing)}"
            )

    def _mark_fresh_optimizer_states_initialized(self) -> None:
        """Tell Megatron's offloader about TE states created by initial install."""

        # Some TE builds eagerly allocate state before this adapter sees it,
        # while others initialize lazily in ``_ensure_adam_state``.  After the
        # authoritative initial install every locally-owned parameter has a
        # complete state either way, so mark every participating optimizer.
        optimizers = {
            id(runtime.dist_optimizer): runtime.dist_optimizer
            for runtime in self.tensors
        }
        marked = 0
        for dist_optimizer in optimizers.values():
            offloader = getattr(dist_optimizer, "_state_offloader", None)
            if offloader is None:
                continue
            marker = getattr(offloader, "mark_optimizer_states_initialized", None)
            if not callable(marker):
                raise RuntimeError(
                    "Megatron optimizer-state offloader has no initialization marker"
                )
            marker()
            if not getattr(offloader, "_optimizer_states_initialized", False):
                raise RuntimeError(
                    "Megatron optimizer-state offloader rejected initialized TE states"
                )
            marked += 1
        self._fresh_state_dist_optimizers.clear()
        if marked and getattr(self, "is_leader", False):
            log.info(
                "marked %d Megatron optimizer offloader(s) initialized after "
                "authoritative TE state install",
                marked,
            )

    @torch.no_grad()
    def _install_local(
        self, runtime: _RuntimeTensor, authoritative: torch.Tensor
    ) -> None:
        self._assert_optimizer_state_resident(runtime)
        if (
            authoritative.dtype != torch.float32
            or tuple(authoritative.shape) != runtime.descriptor.local_shape
        ):
            raise TypeError(
                f"{runtime.descriptor.name}: authoritative shard must be FP32 "
                f"shape {runtime.descriptor.local_shape}, got {authoritative.dtype} "
                f"{tuple(authoritative.shape)}"
            )
        flat = authoritative.contiguous().view(-1)
        if runtime.model_param.dtype == torch.bfloat16:
            rounded_high, remainder = split_fp32_to_bf16_remainder(flat)
            runtime.model_param.copy_(rounded_high.view(runtime.descriptor.local_shape))
        else:
            runtime.model_param.copy_(flat.view(runtime.descriptor.local_shape))

        parameter = runtime.optimizer_param
        if parameter is None:
            return
        assert runtime.owned_start is not None and runtime.owned_end is not None
        start, end = runtime.owned_start, runtime.owned_end
        existing_state = runtime.adam_optimizer.state.get(parameter)
        moment_versions = {
            key: value._version
            for key, value in (existing_state or {}).items()
            if key in ("exp_avg", "exp_avg_sq") and isinstance(value, torch.Tensor)
        }
        if parameter.dtype == torch.bfloat16:
            parameter.copy_(rounded_high[start:end])
            self._ensure_adam_state(runtime)
            runtime.adam_optimizer.set_scaled_state(
                parameter, "master_param", remainder[start:end].contiguous()
            )
            installed_master = runtime.adam_optimizer.get_unscaled_state(
                parameter, "master_param"
            )
            reconstructed = reconstruct_fp32_from_bf16_remainder(
                parameter, installed_master
            )
        else:
            owned = flat[start:end].contiguous()
            parameter.copy_(owned)
            self._ensure_adam_state(runtime)
            runtime.adam_optimizer.set_scaled_state(parameter, "master_param", owned)
            reconstructed = runtime.adam_optimizer.get_unscaled_state(
                parameter, "master_param"
            )

        expected = flat[start:end]
        if not torch.equal(
            reconstructed.contiguous().view(torch.int32),
            expected.contiguous().view(torch.int32),
        ):
            raise RuntimeError(
                f"{runtime.descriptor.name}: installed FP32 optimizer master is not bit-exact"
            )
        current_state = runtime.adam_optimizer.state[parameter]
        for key, version in moment_versions.items():
            value = current_state.get(key)
            if not isinstance(value, torch.Tensor) or value._version != version:
                raise RuntimeError(
                    f"{runtime.descriptor.name}: authoritative install mutated Adam {key}"
                )

    def _gather_one_canonical(self, runtime: _RuntimeTensor) -> torch.Tensor | None:
        """Gather one complete canonical tensor, preserving TP semantics."""

        descriptor = runtime.descriptor
        local = self._local_master(runtime)
        partitions = None
        if descriptor.tp_sharded:
            if self.parallel.cp.rank == 0:
                partitions = [
                    torch.empty_like(local) for _ in range(self.parallel.tp.size)
                ]
                dist.all_gather(partitions, local, group=self.parallel.tp.group)

        def finish_gather() -> torch.Tensor:
            if descriptor.tp_sharded:
                if partitions is None:
                    raise RuntimeError(
                        f"{descriptor.name}: leader has no TP partitions"
                    )
                assert descriptor.partition_dim is not None
                full = self._gather_with_stride(
                    partitions,
                    descriptor.partition_dim,
                    descriptor.partition_stride,
                ).contiguous()
            else:
                full = local
            if full is None or tuple(full.shape) != descriptor.full_shape:
                raise RuntimeError(
                    f"{descriptor.name}: canonical gather produced "
                    f"{None if full is None else tuple(full.shape)}, expected {descriptor.full_shape}"
                )
            return full.cpu()

        result = self._leader_local(finish_gather)
        return result

    def _gather_canonical(self, fid: int) -> torch.Tensor | None:
        """Gather one flat fragment while holding one canonical GPU tensor at a time."""

        fragment = self.layout.fragments[fid]
        spans = self.fragment_spans[fid]
        leader_flat = (
            torch.empty(fragment.numel, dtype=torch.float32, device="cpu")
            if self.is_leader
            else None
        )
        for span in spans:
            full = self._gather_one_canonical(self.tensors_by_name[span.name])

            def validate_and_copy(
                full: torch.Tensor | None = full,
                span: _FragmentTensorSpan = span,
                leader_flat: torch.Tensor | None = leader_flat,
            ) -> None:
                if full is None:
                    raise RuntimeError(f"{span.name}: leader gather returned no tensor")
                if tuple(full.shape) != span.full_shape or full.numel() != span.numel:
                    raise RuntimeError(
                        f"{span.name}: gathered shape {tuple(full.shape)} differs "
                        f"from canonical shape {span.full_shape}"
                    )
                assert leader_flat is not None
                leader_flat[span.start : span.end].copy_(full.reshape(-1))

            self._leader_local(validate_and_copy)
            del full, validate_and_copy
        return leader_flat

    def _apply_one_canonical(
        self,
        runtime: _RuntimeTensor,
        leader_full_cpu: torch.Tensor | None,
        *,
        merge_alpha: float,
    ) -> None:
        """Apply one complete canonical tensor, preserving TP/CP ordering."""

        descriptor = runtime.descriptor
        local = torch.empty(
            descriptor.local_shape, dtype=torch.float32, device=self.device
        )
        if descriptor.tp_sharded:
            def prepare_scatter() -> list[torch.Tensor]:
                if leader_full_cpu is None:
                    raise RuntimeError(
                        f"{descriptor.name}: leader has no canonical tensor"
                    )
                full = leader_full_cpu.to(
                    self.device, non_blocking=False
                ).contiguous()
                assert descriptor.partition_dim is not None
                return [
                    inverse_tp_partition(
                        full,
                        tp_rank=rank,
                        tp_size=self.parallel.tp.size,
                        partition_dim=descriptor.partition_dim,
                        partition_stride=descriptor.partition_stride,
                    )
                    for rank in range(self.parallel.tp.size)
                ]

            scatter_list = self._leader_local(prepare_scatter)
            del prepare_scatter
            if self.parallel.cp.rank == 0:
                dist.scatter(
                    local,
                    scatter_list=scatter_list,
                    src=self.tp_source_global,
                    group=self.parallel.tp.group,
                )
            del scatter_list
            dist.broadcast(
                local,
                src=self.cp_source_global,
                group=self.parallel.cp.group,
            )
        else:
            def prepare_replicated() -> None:
                if leader_full_cpu is None:
                    raise RuntimeError(
                        f"{descriptor.name}: leader has no replicated tensor"
                    )
                local.copy_(leader_full_cpu.to(self.device, non_blocking=False))

            self._leader_local(prepare_replicated)
            del prepare_replicated
            dist.broadcast(local, src=self.leader_global_rank)

        if merge_alpha:
            current = self._local_master(runtime)
            local.mul_(1.0 - merge_alpha).add_(current, alpha=merge_alpha)
        self._install_local(runtime, local)

    def _apply_canonical(
        self,
        fid: int,
        leader_flat_cpu: torch.Tensor | None,
        *,
        merge_alpha: float,
    ) -> None:
        """Apply one flat fragment in layout order, one GPU tensor at a time."""

        fragment = self.layout.fragments[fid]

        def validate_payload() -> None:
            if leader_flat_cpu is None:
                raise RuntimeError(f"fragment {fid}: leader has no canonical payload")
            if leader_flat_cpu.ndim != 1 or leader_flat_cpu.numel() != fragment.numel:
                raise ValueError(
                    f"fragment {fid}: canonical payload has shape "
                    f"{tuple(leader_flat_cpu.shape)}, expected ({fragment.numel},)"
                )

        self._leader_local(validate_payload)
        del validate_payload
        for span in self.fragment_spans[fid]:

            def slice_payload() -> torch.Tensor:
                assert leader_flat_cpu is not None
                return leader_flat_cpu[span.start : span.end].view(
                    span.full_shape
                )

            leader_tensor = self._leader_local(slice_payload)
            del slice_payload
            self._apply_one_canonical(
                self.tensors_by_name[span.name],
                leader_tensor,
                merge_alpha=merge_alpha,
            )
            del leader_tensor

    def _send_initial_parameters(self) -> None:
        # INIT_PARAMS is authoritative only from logical learner 0.  Other
        # islands still connect immediately and wait for that raw global cut.
        if self.learner_id == 0:
            for fid, fragment in enumerate(self.layout.fragments):
                full = self._gather_canonical(fid)

                def send_init(
                    full: torch.Tensor | None = full,
                    fid: int = fid,
                    fragment: Fragment = fragment,
                ) -> None:
                    assert self.client is not None and full is not None
                    payload = _pack_flat_fragment(
                        fragment, full, bulk_dtype(self.client.dtype)
                    )
                    self.client.send_init(
                        fid,
                        payload,
                    )
                    del payload

                self._leader_local(send_init)
                del full, send_init
        dist.barrier()

    def _wait_initial_plan(self) -> list[tuple[int, int]]:
        assert self.client is not None
        deadline = time.monotonic() + self.client.finalization_timeout
        latest: dict[int, Any] = {}
        while len(latest) < self.layout.num_fragments:
            self.client.check_health()
            for update in self.client.drain_updates():
                latest[update.fragment_id] = update
            self.pending_pulls.extend(self.client.drain_pulls())
            if self.client.finalizing.is_set() or self.client.shutdown.is_set():
                raise RuntimeError(
                    "syncer finalized before initial Miles value cut arrived"
                )
            if len(latest) == self.layout.num_fragments:
                break
            if time.monotonic() >= deadline:
                missing = sorted(
                    set(range(self.layout.num_fragments)).difference(latest)
                )
                raise TimeoutError(
                    f"timed out waiting for initial fragments {missing[:16]}"
                )
            time.sleep(0.01)
        metadata = []
        for fid in sorted(latest):
            update = latest[fid]
            self._leader_payloads[(fid, update.version)] = update.data
            metadata.append((fid, update.version))
        return metadata

    def ensure_initial_ready(self) -> None:
        if self.initial_ready:
            return
        metadata = self._leader_value(self._wait_initial_plan)
        for fid, version in metadata:

            def decode_initial(fid: int = fid, version: int = version) -> torch.Tensor:
                assert self.client is not None
                data = self._leader_payloads.pop((fid, version))
                flat = _unpack_flat_fragment(
                    self.layout.fragments[fid], data, bulk_dtype(self.client.dtype)
                )
                self.anchors[fid] = flat.to(torch.bfloat16).contiguous()
                return flat

            full = self._leader_local(decode_initial)
            self._apply_canonical(fid, full, merge_alpha=0.0)
            del full, decode_initial
            self.steps_at_reset[fid] = self.steps_total
            self.units_at_reset[fid] = self.units_total
            self.fragment_versions[fid] = version
        self._mark_fresh_optimizer_states_initialized()
        self.initial_ready = True
        dist.barrier()

    def _normal_update_plan(self) -> tuple[str, list[tuple[int, int]]]:
        assert self.client is not None
        self.client.check_health()
        if self.client.finalizing.is_set():
            return "finalize", []
        if self.client.shutdown.is_set():
            raise RuntimeError("syncer shut down without a final manifest")
        updates = self.client.drain_updates()
        self.pending_pulls.extend(self.client.drain_pulls())
        metadata = []
        for update in sorted(
            updates, key=lambda item: (item.version, item.fragment_id)
        ):
            key = (update.fragment_id, update.version)
            self._leader_payloads[key] = update.data
            metadata.append(key)
        return "continue", metadata

    def _ready_pull_plan(self) -> list[tuple[int, int, int, int, int, int, int]]:
        assert self.client is not None
        self.pending_pulls.extend(self.client.drain_pulls())
        # A pull can be retried while it waits for H local steps.  Retain only
        # the newest attempt so reaching H cannot trigger duplicate gathers
        # and quantization for one syncer round.
        latest: dict[tuple[int, int], Any] = {}
        for pull in self.pending_pulls:
            key = (pull.fragment_id, pull.global_step)
            previous = latest.get(key)
            if previous is None or pull.round_attempt > previous.round_attempt:
                latest[key] = pull
        ready = []
        waiting = []
        for pull in sorted(
            latest.values(),
            key=lambda item: (item.global_step, item.fragment_id, item.round_attempt),
        ):
            fid = pull.fragment_id
            c_steps = self.steps_total - self.steps_at_reset[fid]
            c_units = self.units_total - self.units_at_reset[fid]
            if c_steps < 0 or c_units < 0:
                raise RuntimeError(
                    f"fragment {fid} clock moved backwards: "
                    f"steps={self.steps_total}-{self.steps_at_reset[fid]}, "
                    f"units={self.units_total}-{self.units_at_reset[fid]}"
                )
            if c_steps < self.min_local_steps or self.anchors[fid] is None:
                waiting.append(pull)
                continue
            ready.append(
                (
                    fid,
                    pull.global_step,
                    pull.round_attempt,
                    self.fragment_versions[fid],
                    self.steps_total,
                    c_steps,
                    c_units,
                )
            )
        self.pending_pulls = waiting
        return ready

    def normal_boundary(self) -> None:
        mode, updates = self._leader_value(self._normal_update_plan)
        if mode == "finalize":
            self._finalize_from_manifest()
            return
        for fid, version in updates:

            def decode_update(fid: int = fid, version: int = version) -> torch.Tensor:
                assert self.client is not None
                data = self._leader_payloads.pop((fid, version))
                flat = _unpack_flat_fragment(
                    self.layout.fragments[fid], data, bulk_dtype(self.client.dtype)
                )
                self.anchors[fid] = flat.to(torch.bfloat16).contiguous()
                return flat

            full = self._leader_local(decode_update)
            self._apply_canonical(fid, full, merge_alpha=self.merge_alpha)
            del full, decode_update
            self.steps_at_reset[fid] = self.steps_total
            self.units_at_reset[fid] = self.units_total
            self.fragment_versions[fid] = version

        pulls = self._leader_value(self._ready_pull_plan)
        for (
            fid,
            global_step,
            attempt,
            base_version,
            local_step,
            c_steps,
            c_units,
        ) in pulls:
            full = self._gather_canonical(fid)

            def push_update(
                fid: int = fid,
                global_step: int = global_step,
                attempt: int = attempt,
                base_version: int = base_version,
                local_step: int = local_step,
                c_steps: int = c_steps,
                c_units: int = c_units,
                full: torch.Tensor | None = full,
            ) -> None:
                assert self.client is not None and full is not None
                anchor = self.anchors[fid]
                if anchor is None:
                    raise RuntimeError(f"fragment {fid} has no raw global anchor")
                delta = full.float().sub(anchor.float())
                payload = _quantize_flat_fragment(
                    self.layout.fragments[fid], delta
                )
                del delta
                self.client.push_fragment(
                    fid,
                    global_step,
                    attempt,
                    base_version,
                    local_step,
                    c_steps,
                    c_units,
                    payload,
                )
                del payload

            self._leader_local(push_update)
            del full, push_update

        def heartbeat() -> None:
            assert self.client is not None
            self.client.heartbeat(self.steps_total)

        self._leader_local(heartbeat)

    def after_local_step(self, supervised_tokens: int) -> None:
        if self.finalized:
            raise RuntimeError("Miles attempted a train step after Yeto finalization")
        if isinstance(supervised_tokens, bool) or not isinstance(
            supervised_tokens, int
        ):
            raise TypeError("supervised_tokens must be a plain int")
        if supervised_tokens <= 0:
            raise ValueError(
                f"supervised_tokens must be positive, got {supervised_tokens}"
            )
        self.steps_total += 1
        self.units_total += supervised_tokens
        if self.budget_steps is not None:
            if self.steps_total > self.budget_steps:
                raise RuntimeError(
                    f"local step {self.steps_total} exceeded Yeto budget {self.budget_steps}"
                )
            if self.steps_total == self.budget_steps:
                self._finalize_budget()
                return
        self.normal_boundary()

    def _next_budget_round(self, completed: set[int]) -> tuple[int, int, int, int]:
        assert self.client is not None
        deadline = time.monotonic() + self.client.finalization_timeout
        bases: dict[int, Any] = getattr(self, "_budget_bases", {})
        pulls: list[Any] = getattr(self, "_budget_pulls", [])
        self._budget_bases = bases
        self._budget_pulls = pulls
        while True:
            self.client.check_health()
            for update in self.client.drain_updates():
                if update.fragment_id not in completed:
                    bases[update.fragment_id] = update
            pulls.extend(self.client.drain_pulls())
            eligible = [
                pull
                for pull in pulls
                if pull.fragment_id not in completed and pull.fragment_id in bases
            ]
            if eligible:
                eligible.sort(
                    key=lambda item: (
                        item.global_step,
                        item.fragment_id,
                        item.round_attempt,
                    )
                )
                pull = eligible[0]
                pulls.remove(pull)
                base = bases.pop(pull.fragment_id)
                self._leader_payloads[(base.fragment_id, base.version)] = base.data
                return (
                    pull.fragment_id,
                    pull.global_step,
                    pull.round_attempt,
                    base.version,
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "budget consolidation timed out waiting for pull/base"
                )
            time.sleep(0.01)

    def _finalize_budget(self) -> None:
        dist.barrier()

        def restart() -> bool:
            assert self.client is not None
            generation = self.client.send_budget_done(self.steps_total)
            self.client.wait_for_budget_restart(generation)
            return True

        self._leader_value(restart)
        completed: set[int] = set()
        for _ in range(self.layout.num_fragments):
            fid, global_step, attempt, base_version = self._leader_value(
                lambda: self._next_budget_round(completed)
            )
            frozen = self._gather_canonical(fid)

            def push_frozen(
                fid: int = fid,
                global_step: int = global_step,
                attempt: int = attempt,
                base_version: int = base_version,
                frozen: torch.Tensor | None = frozen,
            ) -> None:
                assert self.client is not None and frozen is not None
                data = self._leader_payloads.pop((fid, base_version))
                base = _unpack_flat_fragment(
                    self.layout.fragments[fid], data, bulk_dtype(self.client.dtype)
                )
                delta = frozen.float().sub(base.float())
                del base
                payload = _quantize_flat_fragment(
                    self.layout.fragments[fid], delta
                )
                del delta
                self.client.push_fragment(
                    fid,
                    global_step,
                    attempt,
                    base_version,
                    self.steps_total,
                    self.steps_total,
                    self.units_total,
                    payload,
                )
                del payload

            self._leader_local(push_frozen)
            del frozen, push_frozen
            completed.add(fid)
        self._finalize_from_manifest()

    def _finalize_from_manifest(self) -> None:
        def wait_manifest() -> tuple[int, tuple[int, ...]]:
            assert self.client is not None
            manifest, updates = self.client.wait_for_final_fragments()
            by_fid = {update.fragment_id: update for update in updates}
            if set(by_fid) != set(range(self.layout.num_fragments)):
                raise RuntimeError("terminal manifest is missing Miles value fragments")
            for fid, expected_version in enumerate(manifest.versions):
                update = by_fid[fid]
                if update.version != expected_version:
                    raise RuntimeError(
                        f"terminal fragment {fid} version {update.version} != {expected_version}"
                    )
                self._leader_payloads[(fid, update.version)] = update.data
            return manifest.global_step, manifest.versions

        global_step, versions = self._leader_value(wait_manifest)
        for fid, version in enumerate(versions):

            def decode_terminal(fid: int = fid, version: int = version) -> torch.Tensor:
                data = self._leader_payloads.pop((fid, version))
                return _unpack_flat_fragment(
                    self.layout.fragments[fid], data, DTYPE_F32
                )

            full = self._leader_local(decode_terminal)
            self._apply_canonical(fid, full, merge_alpha=0.0)
            del full, decode_terminal
            self.fragment_versions[fid] = version
        dist.barrier()

        def acknowledge() -> bool:
            assert self.client is not None
            from ..protocol import FinalManifest

            self.client.acknowledge_finalization(
                FinalManifest(global_step, tuple(versions))
            )
            return True

        self._leader_value(acknowledge)
        dist.barrier()
        self.finalized = True
        if self.is_leader:
            assert self.client is not None
            self.client.close()
            log.info(
                "installed authoritative Yeto final cut global_step=%d local_steps=%d units=%d",
                global_step,
                self.steps_total,
                self.units_total,
            )


_ISLAND: MilesValueIsland | None = None


def after_model_init(
    args: Any,
    role: str,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Miles hook: construct/connect the critic island after checkpoint load."""

    del opt_param_scheduler
    global _ISLAND
    if role != "critic":
        return
    if _ISLAND is not None:
        raise RuntimeError("Miles value island was initialized twice in one process")
    _install_bf16_static_unscale_compat(args, optimizer)
    _install_mixed_dtype_grad_compat(args)
    _ISLAND = MilesValueIsland(args, model, optimizer)


def before_train_step(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Miles hook: install the authoritative initial cut before first FWD."""

    del rollout_id, step_id, model, optimizer, opt_param_scheduler
    if getattr(args, "debug_disable_optimizer", False):
        return
    if _ISLAND is None:
        raise RuntimeError(
            "Miles value island before-step hook ran before initialization"
        )
    _ISLAND.ensure_initial_ready()


def after_train_step(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: Any,
    optimizer: Any,
    opt_param_scheduler: Any,
    supervised_tokens: int,
) -> None:
    """Miles hook: run one all-rank Yeto boundary after a successful step."""

    del rollout_id, step_id, model, optimizer, opt_param_scheduler
    if getattr(args, "debug_disable_optimizer", False):
        raise RuntimeError(
            "Miles called value-island after-step hook during validation"
        )
    if _ISLAND is None:
        raise RuntimeError(
            "Miles value island after-step hook ran before initialization"
        )
    _ISLAND.after_local_step(supervised_tokens)


__all__ = [
    "MilesValueIsland",
    "TensorDescriptor",
    "after_model_init",
    "after_train_step",
    "before_train_step",
    "grouped_tensor_fragment_layout",
    "one_tensor_fragment_layout",
]
