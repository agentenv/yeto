"""Correctness-gated causal-LM attention and training-kernel policies.

The native path is always available and remains the default. Optional CUDA
kernels are selected explicitly, pinned to versions exercised by the A100
benchmark, and rejected before model loading when their contract cannot be
met. This module deliberately contains no silent fallback: an explicit
request either resolves exactly or raises an actionable error.
"""

from __future__ import annotations

import importlib.util
import inspect
from importlib import metadata

import torch

from .kernel_deps import FLASH_ATTN_VERSION, LIGER_KERNEL_VERSION

ATTENTION_BACKENDS = ("auto", "sdpa", "flash-attn-2")
KERNEL_BACKENDS = ("native", "liger")

_HF_ATTENTION_NAMES = {
    "sdpa": "sdpa",
    "flash-attn-2": "flash_attention_2",
}
_DISPLAY_ATTENTION_NAMES = {value: key for key, value in _HF_ATTENTION_NAMES.items()}
_PACKAGE_INSTALL_HINTS = {
    "liger-kernel": "pip install -e '.[a100-liger]'",
    "flash-attn": "the pinned --no-build-isolation command in docs/A100_KERNELS.md",
}


def _require_a100(device) -> None:
    try:
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
    except Exception as exc:
        raise RuntimeError("could not verify the CUDA device for the A100 kernel lane") from exc
    if "A100" not in name or capability != (8, 0):
        raise RuntimeError(
            f"the optimized kernel lane is scoped to A100/SM80; found "
            f"{name!r} with capability {capability}"
        )


def _require_exact_package(distribution: str, import_name: str, required: str) -> None:
    try:
        installed = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        hint = _PACKAGE_INSTALL_HINTS.get(distribution, f"install {distribution}=={required}")
        raise RuntimeError(
            f"{distribution}=={required} is required; use {hint} on the A100 node"
        ) from exc
    if installed != required:
        raise RuntimeError(
            f"{distribution}=={required} is required, but version {installed} is installed"
        )
    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(
            f"{distribution}=={required} is installed but {import_name!r} is not importable"
        )


def attention_load_kwargs(requested: str, device, dtype: torch.dtype) -> dict[str, str]:
    """Map a public attention selection to Hugging Face load kwargs."""
    if requested not in ATTENTION_BACKENDS:
        raise ValueError(
            f"unknown attention backend {requested!r}; choose from {ATTENTION_BACKENDS}"
        )
    if requested == "auto":
        return {}
    if requested == "flash-attn-2":
        if device.type != "cuda":
            raise RuntimeError("--attention-backend flash-attn-2 requires CUDA")
        if dtype is not torch.bfloat16:
            raise RuntimeError(
                "--attention-backend flash-attn-2 requires BF16 in the A100 lane; "
                "use --attention-backend sdpa for FP32"
            )
        _require_a100(device)
        _require_exact_package("flash-attn", "flash_attn", FLASH_ATTN_VERSION)
    return {"attn_implementation": _HF_ATTENTION_NAMES[requested]}


def resolved_attention_backend(model, requested: str) -> str:
    """Read and verify the attention implementation selected by Transformers."""
    config = getattr(model, "config", None)
    resolved = getattr(config, "_attn_implementation", None)
    if resolved is None:
        resolved = getattr(config, "attn_implementation", None)
    if isinstance(resolved, dict):
        normalized = {
            key: _DISPLAY_ATTENTION_NAMES.get(value, value or "unknown")
            for key, value in resolved.items()
        }
        resolved_values = set(normalized.values())
        display = str(normalized)
    else:
        display = _DISPLAY_ATTENTION_NAMES.get(resolved, resolved or "unknown")
        resolved_values = {display}
    if requested != "auto" and resolved_values != {requested}:
        raise RuntimeError(
            f"requested attention backend {requested!r}, but the loaded model resolved {display!r}"
        )
    return str(display)


def validate_kernel_request(
    kernel_backend: str,
    loss_function: str,
    device,
    dtype: torch.dtype,
    base_quantization: str = "none",
) -> None:
    """Reject unsupported optimized-kernel combinations before model loading."""
    if kernel_backend not in KERNEL_BACKENDS:
        raise ValueError(f"unknown kernel backend {kernel_backend!r}; choose from {KERNEL_BACKENDS}")
    if kernel_backend == "native":
        return
    if device.type != "cuda":
        raise RuntimeError("--kernel-backend liger requires CUDA")
    _require_a100(device)
    if dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("the A100 Liger lane supports BF16 and FP32 only")
    if loss_function != "cross_entropy":
        raise ValueError(
            "--kernel-backend liger supports only the built-in cross_entropy loss; "
            "use --kernel-backend native for custom, pickled, or RL losses"
        )
    if base_quantization != "none":
        raise ValueError(
            "--kernel-backend liger does not support a quantized base; "
            "use --base-quantization none or --kernel-backend native"
        )
    _require_exact_package("liger-kernel", "liger_kernel", LIGER_KERNEL_VERSION)


def require_liger_model_support(config) -> str:
    """Ensure Liger 0.8.0 can provide fused linear CE for this model type."""
    try:
        from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
    except Exception as exc:
        raise RuntimeError("could not import the pinned Liger model registry") from exc

    model_type = getattr(config, "model_type", None)
    apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(model_type)
    if apply_fn is None:
        raise RuntimeError(
            f"Liger {LIGER_KERNEL_VERSION} does not support model type {model_type!r}"
        )
    if "fused_linear_cross_entropy" not in inspect.signature(apply_fn).parameters:
        raise RuntimeError(
            f"Liger {LIGER_KERNEL_VERSION} has no fused linear CE path for "
            f"model type {model_type!r}"
        )
    # Liger registers Granite for other kernels but explicitly rejects fused
    # linear CE in this pinned release.
    if model_type == "granite":
        raise RuntimeError(
            f"Liger {LIGER_KERNEL_VERSION} does not implement fused linear CE for Granite"
        )
    return str(model_type)


def binary_mask_labels(
    input_ids: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a binary per-token weight mask to labels with ignore_index."""
    if input_ids.shape != weights.shape:
        raise ValueError(
            f"input IDs and weights must have the same shape, got {input_ids.shape} and {weights.shape}"
        )
    is_binary = torch.logical_or(weights == 0, weights == 1)
    if not bool(is_binary.all().item()):
        raise ValueError(
            "--kernel-backend liger requires binary 0/1 token weights; "
            "fractional weights require --kernel-backend native"
        )
    labels = input_ids.masked_fill(weights != 1, -100)
    target_tokens = (labels[:, 1:] != -100).sum()
    return labels, target_tokens


def liger_sft_forward(model, input_ids: torch.Tensor, weights: torch.Tensor):
    """Run Liger fused linear CE and return a local token-sum loss."""
    labels, target_tokens = binary_mask_labels(input_ids, weights)
    output = model(
        input_ids=input_ids,
        labels=labels,
        num_items_in_batch=1,
        use_cache=False,
    )
    loss = getattr(output, "loss", None)
    if loss is None:
        raise RuntimeError("the Liger model returned no fused cross-entropy loss")
    if loss.ndim != 0:
        raise RuntimeError(f"the Liger model returned a non-scalar loss with shape {loss.shape}")
    if getattr(output, "logits", None) is not None:
        raise RuntimeError(
            "the requested Liger fused-loss path materialized logits; "
            "this model/version combination is unsupported"
        )
    # Liger 0.8.0 selects reduction='sum' when num_items_in_batch is given,
    # then divides by that value. Passing one therefore preserves Yeto's
    # local token-SUM contract exactly.
    return loss, target_tokens
