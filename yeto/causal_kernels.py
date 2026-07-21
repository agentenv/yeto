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

NATIVE_LAYER_BACKEND = "transformers-native"
NATIVE_LOSS_IMPLEMENTATION = "torch-cross-entropy"
FUSED_LINEAR_CE_IMPLEMENTATION = "liger-fused-linear-cross-entropy"

_QWEN2_APPLY_CONTROLS = (
    ("rope", True),
    ("cross_entropy", False),
    ("fused_linear_cross_entropy", True),
    ("rms_norm", True),
    ("swiglu", True),
    ("model", None),
)

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
        raise RuntimeError("--kernel-backend liger fused-linear-CE requires CUDA")
    _require_a100(device)
    if dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError("the A100 Liger fused-linear-CE lane supports BF16 and FP32 only")
    if loss_function != "cross_entropy":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE supports only the built-in "
            "cross_entropy loss; "
            "use --kernel-backend native for custom, pickled, or RL losses"
        )
    if base_quantization != "none":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE does not support a quantized base; "
            "use --base-quantization none or --kernel-backend native"
        )
    _require_exact_package("liger-kernel", "liger_kernel", LIGER_KERNEL_VERSION)


def _qwen2_fused_linear_ce_apply_function():
    """Return the pinned public Qwen2 apply function after an exact ABI check."""
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        from liger_kernel.transformers import functional as liger_functional
    except Exception as exc:
        raise RuntimeError(
            "could not import Liger's public Qwen2 kernel apply function"
        ) from exc

    signature = inspect.signature(apply_liger_kernel_to_qwen2)
    parameters = tuple(signature.parameters.values())
    expected_names = tuple(name for name, _default in _QWEN2_APPLY_CONTROLS)
    if tuple(parameter.name for parameter in parameters) != expected_names:
        raise RuntimeError(
            f"Liger {LIGER_KERNEL_VERSION}'s Qwen2 apply function does not expose "
            f"the exact controls {expected_names}; found {tuple(signature.parameters)}"
        )
    for parameter, (_name, expected_default) in zip(
        parameters, _QWEN2_APPLY_CONTROLS, strict=True
    ):
        if parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            raise RuntimeError(
                f"Liger {LIGER_KERNEL_VERSION}'s Qwen2 control "
                f"{parameter.name!r} has unsupported kind {parameter.kind}"
            )
        if parameter.default != expected_default:
            raise RuntimeError(
                f"Liger {LIGER_KERNEL_VERSION}'s Qwen2 control "
                f"{parameter.name!r} has unexpected default {parameter.default!r}; "
                f"expected {expected_default!r}"
            )
    fused_signature = inspect.signature(
        liger_functional.liger_fused_linear_cross_entropy
    )
    accum_dtype = fused_signature.parameters.get("accum_dtype")
    if accum_dtype is None or accum_dtype.kind not in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        raise RuntimeError(
            f"Liger {LIGER_KERNEL_VERSION}'s fused-linear-CE primitive does not "
            "explicitly support accum_dtype; refusing to let FP32 accumulation "
            "be silently filtered"
        )
    return apply_liger_kernel_to_qwen2


def require_liger_model_support(config) -> str:
    """Require the one model family approved for isolated fused linear CE."""

    model_type = getattr(config, "model_type", None)
    if model_type != "qwen2":
        raise RuntimeError(
            "the isolated Liger fused-linear-CE lane supports only Hugging Face "
            f"Qwen2/Qwen2.5 (model_type='qwen2'); found {model_type!r}"
        )
    _qwen2_fused_linear_ce_apply_function()
    return str(model_type)


def _namespace_is_identical(before: dict, after: dict) -> bool:
    """Compare a shallow namespace without invoking user equality methods."""
    return before.keys() == after.keys() and all(
        after[name] is value for name, value in before.items()
    )


def apply_liger_fused_linear_ce(model) -> dict:
    """Patch only one loaded Qwen2 model instance's loss-producing forward.

    Every layer kernel remains the Transformers implementation.  The strict
    pre/post snapshots turn the third-party apply call into a fail-closed
    operation: the only permitted state change is a new ``forward`` binding on
    this exact model instance.  In particular, the Qwen2 module, model class,
    functional cross entropy, and every nested module must remain untouched.
    """
    model_type = require_liger_model_support(getattr(model, "config", None))
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except Exception as exc:
        raise RuntimeError("could not import the pinned Transformers Qwen2 model") from exc

    expected_class = modeling_qwen2.Qwen2ForCausalLM
    if type(model) is not expected_class:
        raise RuntimeError(
            "the isolated fused-linear-CE patch must run on an unwrapped, native "
            f"Qwen2ForCausalLM instance before PEFT; found {type(model)!r}"
        )
    if "forward" in vars(model):
        raise RuntimeError(
            "the isolated fused-linear-CE patch requires an unmodified model "
            "instance with no pre-existing forward override"
        )

    apply_fn = _qwen2_fused_linear_ce_apply_function()
    module_globals_before = dict(vars(modeling_qwen2))
    class_globals_before = dict(vars(expected_class))
    functional_ce_before = torch.nn.functional.cross_entropy
    instance_before = dict(vars(model))
    nested_before = [
        (name, module, dict(vars(module)))
        for name, module in model.named_modules()
        if module is not model
    ]
    inherited_forward = model.forward

    apply_fn(
        rope=False,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
        rms_norm=False,
        swiglu=False,
        model=model,
    )

    module_globals_unchanged = _namespace_is_identical(
        module_globals_before, dict(vars(modeling_qwen2))
    )
    class_globals_unchanged = _namespace_is_identical(
        class_globals_before, dict(vars(expected_class))
    )
    functional_ce_unchanged = (
        torch.nn.functional.cross_entropy is functional_ce_before
    )
    current_modules = list(model.named_modules())
    expected_modules = [("", model)] + [
        (name, module) for name, module, _state in nested_before
    ]
    module_layout_unchanged = len(current_modules) == len(expected_modules) and all(
        actual_name == expected_name and actual_module is expected_module
        for (actual_name, actual_module), (expected_name, expected_module) in zip(
            current_modules, expected_modules, strict=True
        )
    )
    nested_state_unchanged = module_layout_unchanged and all(
        _namespace_is_identical(state, dict(vars(module)))
        for _name, module, state in nested_before
    )
    instance_after = dict(vars(model))
    instance_change_is_forward_only = (
        instance_after.keys() == instance_before.keys() | {"forward"}
        and all(instance_after[name] is value for name, value in instance_before.items())
    )
    patched_forward = instance_after.get("forward")
    forward_is_instance_bound = (
        patched_forward is not None
        and getattr(patched_forward, "__self__", None) is model
        and getattr(patched_forward, "__func__", None)
        is not getattr(inherited_forward, "__func__", None)
    )
    fused_forward_keyword_contract = False
    if callable(patched_forward):
        patched_signature = inspect.signature(patched_forward)
        has_forward_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in patched_signature.parameters.values()
        )
        fused_forward_keyword_contract = (
            {"labels", "use_cache", "skip_logits"}
            <= patched_signature.parameters.keys()
            and has_forward_kwargs
        )

    invariants = {
        "qwen2_module_globals_unchanged": module_globals_unchanged,
        "qwen2_class_globals_unchanged": class_globals_unchanged,
        "torch_cross_entropy_global_unchanged": functional_ce_unchanged,
        "nested_module_layout_unchanged": module_layout_unchanged,
        "nested_module_state_unchanged": nested_state_unchanged,
        "instance_change_is_forward_only": instance_change_is_forward_only,
        "forward_is_instance_bound": forward_is_instance_bound,
        "fused_forward_keyword_contract": fused_forward_keyword_contract,
    }
    failed_invariants = [name for name, passed in invariants.items() if not passed]
    if failed_invariants:
        raise RuntimeError(
            "the isolated fused-linear-CE apply call changed state outside the "
            f"approved model-instance forward binding: {failed_invariants}"
        )

    return {
        "model_type": model_type,
        "layer_backend": NATIVE_LAYER_BACKEND,
        "loss_backend": "liger",
        "loss_implementation": FUSED_LINEAR_CE_IMPLEMENTATION,
        "patch_scope": "model-instance-forward-only",
        "apply_function": (
            "liger_kernel.transformers.apply_liger_kernel_to_qwen2"
        ),
        "apply_controls": {
            "rope": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
            "rms_norm": False,
            "swiglu": False,
        },
        "invariants": invariants,
    }


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
            "--kernel-backend liger fused-linear-CE requires binary 0/1 token weights; "
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
        accum_dtype=torch.float32,
        skip_logits=True,
        use_cache=False,
    )
    loss = getattr(output, "loss", None)
    if loss is None:
        raise RuntimeError("the fused-linear-CE forward returned no loss")
    if loss.ndim != 0:
        raise RuntimeError(
            f"the fused-linear-CE forward returned a non-scalar loss with shape {loss.shape}"
        )
    if getattr(output, "logits", None) is not None:
        raise RuntimeError(
            "the requested Liger fused-loss path materialized logits; "
            "this model/version combination is unsupported"
        )
    # Liger 0.8.0 selects reduction='sum' when num_items_in_batch is given,
    # then divides by that value. Passing one therefore preserves Yeto's
    # local token-SUM contract exactly.
    return loss, target_tokens
