"""Bounded-memory gradient transfer for Megatron's CPU-offloaded AdamW.

Megatron's :class:`HybridDeviceOptimizer` normally copies every GPU gradient
to a distinct pinned FP32 CPU tensor before calling AdamW.  At full offload
that retains one model-sized FP32 gradient set in addition to the FP32 master
and two FP32 Adam moments.  The diagnostic helper in this module keeps the
same HDO and AdamW instances, but presents one parameter at a time to AdamW
through one reusable pinned FP32 scratch allocation.

The implementation is deliberately installed only after checkpoint loading
and before the first optimizer step.  It refuses pre-existing Adam state,
overlapped HDO, partial offload, GPU optimizer shards, closures, and unknown
optimizer hooks.  Those restrictions keep the narrow TP4/CP1 diagnostic path
fail-closed instead of silently changing optimizer semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from types import MethodType
from typing import Any

import torch

_SUPPORTED_GRAD_DTYPES = (torch.bfloat16, torch.float32)


def _ordered_params(optimizer: Any) -> list[torch.Tensor]:
    return [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]


def _source_gradient(hdo: Any, cpu_parameter: torch.Tensor) -> torch.Tensor | None:
    try:
        model_parameter = hdo.cpu_copys_map_gpu_param[cpu_parameter]
    except KeyError as exc:
        raise RuntimeError(
            "streamed HDO CPU parameter has no source-model mapping"
        ) from exc
    gradient = getattr(model_parameter, "decoupled_grad", None)
    if gradient is None:
        gradient = model_parameter.grad
    return gradient


def _copy_gradient(hdo: Any, destination: torch.Tensor, source: torch.Tensor) -> None:
    _validate_source_gradient(hdo, destination.numel(), source)
    if getattr(hdo, "_yeto_require_cuda_gradients", True):
        hdo._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(hdo._d2h_stream):
            destination.copy_(source, non_blocking=True)
        hdo._d2h_stream.record_event().synchronize()
        return
    destination.copy_(source, non_blocking=False)


def _validate_source_gradient(
    hdo: Any, expected_numel: int, source: torch.Tensor
) -> None:
    if source.dtype not in _SUPPORTED_GRAD_DTYPES:
        raise TypeError(
            f"streamed HDO gradient must be BF16 or FP32, got {source.dtype}"
        )
    if source.numel() != expected_numel:
        raise RuntimeError(
            f"streamed HDO gradient size mismatch: {source.numel()} != {expected_numel}"
        )
    if (
        getattr(hdo, "_yeto_require_cuda_gradients", True)
        and source.device.type != "cuda"
    ):
        raise RuntimeError(
            "production streamed HDO requires CUDA source gradients, got "
            f"{source.device}"
        )


def _stream_cpu_adam_step(
    cpu_optimizer: torch.optim.Optimizer,
    closure: Callable[[], float] | None = None,
) -> None:
    """Run one logical AdamW step with one resident CPU gradient at a time."""

    if closure is not None:
        raise RuntimeError("streamed HDO diagnostic does not support closures")
    hdo = cpu_optimizer._yeto_hdo_owner
    original_step = cpu_optimizer._yeto_original_step
    original_groups = list(cpu_optimizer.param_groups)
    group_params = [(group, list(group["params"])) for group in original_groups]

    # Stock dense HDO retains each CPU grad allocation between steps. If a
    # later source gradient is missing, its bulk path leaves the old value in
    # ``param.grad`` and silently applies that stale gradient again. Every
    # trainable parameter in this dense Qwen diagnostic must have a gradient,
    # so preflight the entire source set before changing any master/moment.
    # This also prevents a malformed late parameter from causing a partial
    # logical optimizer step.
    ordered_gradients: list[torch.Tensor] = []
    for _, parameters in group_params:
        for position, cpu_parameter in enumerate(parameters):
            gradient = _source_gradient(hdo, cpu_parameter)
            if gradient is None:
                raise RuntimeError(
                    "streamed HDO source gradient is missing; refusing "
                    f"stale-gradient reuse (group position {position})"
                )
            _validate_source_gradient(hdo, cpu_parameter.numel(), gradient)
            ordered_gradients.append(gradient)

    try:
        gradient_index = 0
        for group, parameters in group_params:
            for cpu_parameter in parameters:
                gradient = ordered_gradients[gradient_index]
                gradient_index += 1

                scratch = hdo._yeto_cpu_gradient_scratch
                scratch_view = scratch[: cpu_parameter.numel()].view_as(cpu_parameter)
                _copy_gradient(hdo, scratch_view, gradient)
                cpu_parameter.requires_grad_(False)
                cpu_parameter.grad = scratch_view

                # Keep the original AdamW object and its wrapped step method.
                # Its one HDO post-hook therefore sees only this active
                # parameter and performs the normal FP32-master -> BF16-model
                # copy before the scratch storage is reused.
                group["params"] = [cpu_parameter]
                cpu_optimizer.param_groups = [group]
                original_step()
                cpu_parameter.grad = None
    finally:
        for group, parameters in group_params:
            group["params"] = parameters
        cpu_optimizer.param_groups = original_groups
        for parameter in _ordered_params(cpu_optimizer):
            parameter.grad = None


def _skip_bulk_gradient_copy(hdo: Any) -> None:
    """Replace HDO's model-sized D2H staging pass with the streamed step."""

    if hdo._cpu_optimizer_map_data_event:
        raise RuntimeError("streamed HDO found an unexpected pending D2H event")


def install_hdo_cpu_gradient_streaming(
    hdo: Any,
    *,
    require_cuda_gradients: bool = True,
) -> int:
    """Install the fresh-run, full-offload HDO streaming diagnostic.

    Returns the persistent scratch size in bytes.  No master parameter or
    optimizer-state tensor is replaced; only bulk gradient staging is changed.
    """

    if getattr(hdo, "_yeto_cpu_gradient_streaming", False):
        raise RuntimeError("HDO CPU-gradient streaming was installed twice")
    if float(getattr(hdo, "offload_fraction", -1.0)) != 1.0:
        raise RuntimeError("streamed HDO requires offload_fraction=1.0")
    if not getattr(hdo, "param_update_in_fp32", False):
        raise RuntimeError("streamed HDO requires FP32 parameter updates")
    if getattr(hdo, "overlap_cpu_optimizer_d2h_h2d", True):
        raise RuntimeError("streamed HDO requires non-overlapped CPU AdamW")
    if getattr(hdo, "gpu_optimizer", None) is not None:
        raise RuntimeError("streamed HDO requires every optimizer shard on CPU")

    cpu_optimizers = list(getattr(hdo, "cpu_optimizers", ()))
    if len(cpu_optimizers) != 1:
        raise RuntimeError(
            "streamed HDO requires exactly one non-overlapped CPU optimizer, "
            f"got {len(cpu_optimizers)}"
        )
    cpu_optimizer = cpu_optimizers[0]
    if not isinstance(cpu_optimizer, torch.optim.AdamW):
        raise TypeError(
            "streamed HDO supports torch.optim.AdamW only, got "
            f"{type(cpu_optimizer).__name__}"
        )
    if cpu_optimizer.state:
        raise RuntimeError(
            "streamed HDO must be installed before Adam state initialization"
        )
    if getattr(hdo, "cpu_copy_map_grad", None):
        raise RuntimeError("streamed HDO found retained CPU gradient tensors")
    if getattr(hdo, "_cpu_optimizer_map_data_event", None):
        raise RuntimeError("streamed HDO found pending CPU optimizer events")

    pre_hooks = getattr(cpu_optimizer, "_optimizer_step_pre_hooks", None)
    post_hooks = getattr(cpu_optimizer, "_optimizer_step_post_hooks", None)
    if pre_hooks is not None and len(pre_hooks) != 0:
        raise RuntimeError("streamed HDO refuses unknown AdamW pre-step hooks")
    # HybridDeviceOptimizer registers exactly one post-hook that copies the
    # updated CPU parameter back to its original GPU parameter.
    if post_hooks is not None and len(post_hooks) != 1:
        raise RuntimeError(
            "streamed HDO expected exactly one AdamW copy-back hook, got "
            f"{len(post_hooks)}"
        )

    parameters = _ordered_params(cpu_optimizer)
    if not parameters:
        raise RuntimeError("streamed HDO found no CPU parameters")
    if any(parameter.device.type != "cpu" for parameter in parameters):
        raise RuntimeError("streamed HDO inner parameters must all reside on CPU")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise TypeError("streamed HDO inner parameters must all be FP32")
    if len(set(parameters)) != len(parameters):
        raise RuntimeError("streamed HDO CPU parameter list contains duplicates")
    if set(parameters) != set(hdo.cpu_copys_map_gpu_param):
        raise RuntimeError("streamed HDO parameter/source mapping is not bijective")
    source_parameters = list(hdo.cpu_copys_map_gpu_param.values())
    if len(set(source_parameters)) != len(source_parameters):
        raise RuntimeError("streamed HDO source-parameter mapping contains duplicates")
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("streamed HDO must be installed before gradients exist")

    max_numel = max(parameter.numel() for parameter in parameters)
    pin_memory = bool(getattr(hdo, "pin_cpu_grads", True))
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False
    scratch = torch.empty(
        max_numel,
        dtype=torch.float32,
        device="cpu",
        pin_memory=pin_memory,
    )

    cpu_optimizer._yeto_hdo_owner = hdo
    cpu_optimizer._yeto_original_step = cpu_optimizer.step
    cpu_optimizer.step = MethodType(_stream_cpu_adam_step, cpu_optimizer)
    hdo._set_sub_optimizer_grads = MethodType(_skip_bulk_gradient_copy, hdo)
    hdo._yeto_cpu_gradient_scratch = scratch
    hdo._yeto_require_cuda_gradients = require_cuda_gradients
    hdo._yeto_cpu_gradient_streaming = True
    hdo._yeto_bulk_gradient_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in parameters
    )
    return scratch.numel() * scratch.element_size()


__all__ = ["install_hdo_cpu_gradient_streaming"]
