#!/usr/bin/env python3
"""Parity-gated causal-LM kernel benchmark for one 8xA100 node.

This script never provisions infrastructure. Run it on an existing node with
``torchrun --standalone --nproc_per_node=8``. Optional combinations that are
not installed are recorded as skipped; a parity failure is fatal and is never
converted into a performance result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from yeto.causal_kernels import (  # noqa: E402
    FUSED_LINEAR_CE_IMPLEMENTATION,
    KernelIsolationError,
    LIGER_QWEN2_SOURCE_SHA256,
    NATIVE_LAYER_BACKEND,
    NATIVE_LOSS_IMPLEMENTATION,
    PEFT_VERSION,
    apply_liger_fused_linear_ce,
    attention_load_kwargs,
    liger_sft_forward,
    require_liger_model_support,
    resolved_attention_backend,
    validate_kernel_request,
    validate_lora_production_envelope,
)
from yeto.losses import sft_loss  # noqa: E402
from yeto.learner import allreduce_trainable_grads, resolve_lora_targets  # noqa: E402
from yeto.models import resolve  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    attention_backend: str
    layer_backend: str
    loss_backend: str
    loss_implementation: str


VARIANTS = (
    Variant(
        "native-sdpa",
        "sdpa",
        NATIVE_LAYER_BACKEND,
        "native",
        NATIVE_LOSS_IMPLEMENTATION,
    ),
    Variant(
        "native-flash-attn-2",
        "flash-attn-2",
        NATIVE_LAYER_BACKEND,
        "native",
        NATIVE_LOSS_IMPLEMENTATION,
    ),
    Variant(
        "fused-linear-ce-sdpa",
        "sdpa",
        NATIVE_LAYER_BACKEND,
        "liger",
        FUSED_LINEAR_CE_IMPLEMENTATION,
    ),
)
VARIANTS_BY_NAME = {variant.name: variant for variant in VARIANTS}
REFERENCE_VARIANT = VARIANTS[0]


def percentile(values: list[float], quantile: float) -> float:
    """Linearly interpolated percentile without an optional NumPy dependency."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def select_variants(spec: str) -> list[Variant]:
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if not names or names == ["all"]:
        return list(VARIANTS)
    unknown = [name for name in names if name not in VARIANTS_BY_NAME]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {list(VARIANTS_BY_NAME)}")
    selected = set(names)
    selected.add(REFERENCE_VARIANT.name)  # every result needs the same parity anchor
    return [variant for variant in VARIANTS if variant.name in selected]


def setup_distributed() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("the A100 kernel benchmark requires CUDA")
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local_rank = 0, 1, 0
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    if "A100" not in name or capability != (8, 0):
        raise RuntimeError(
            f"this benchmark is scoped to A100 (SM80), found {name!r} with capability {capability}"
        )
    return rank, world, device


def distributed_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def distributed_sum(value: int, device: torch.device) -> int:
    tensor = torch.tensor(value, dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def all_ranks_succeeded(succeeded: bool, device: torch.device) -> bool:
    flag = torch.tensor(1 if succeeded else 0, dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def any_rank_true(value: bool, device: torch.device) -> bool:
    flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def gather_errors(error: str | None, world: int) -> list[str]:
    if not dist.is_initialized():
        return [error] if error else []
    errors: list[str | None] = [None] * world
    dist.all_gather_object(errors, error)
    return [item for item in errors if item]


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def is_fatal_model_load_error(exc: Exception) -> bool:
    """Return whether a failed arm makes continuing in-process unsafe."""
    pending: list[BaseException] = [exc]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        try:
            if "out of memory" in str(current).lower():
                return True
        except Exception:
            pass
        if isinstance(current, KernelIsolationError) and current.fatal:
            return True
        for chained in (current.__cause__, current.__context__):
            if chained is not None:
                pending.append(chained)
    return False


def resolve_model_revision(model_id: str, requested_revision: str | None) -> str:
    """Resolve a moving Hub revision to the immutable commit loaded by every rank."""
    if Path(model_id).exists():
        raise ValueError(
            "the reproducibility benchmark requires a Hub model ID; local model "
            "paths do not provide an independently resolvable commit SHA"
        )
    from huggingface_hub import HfApi

    requested = requested_revision or "main"
    try:
        info = HfApi().model_info(model_id, revision=requested)
    except Exception as exc:
        # An explicit full SHA is already immutable and remains usable from a
        # warm offline cache even when the metadata endpoint is unavailable.
        if requested_revision and len(requested_revision) == 40 and all(
            character in "0123456789abcdefABCDEF" for character in requested_revision
        ):
            return requested_revision.lower()
        raise RuntimeError(
            f"could not resolve model {model_id!r} revision {requested!r} to a Hub commit"
        ) from exc
    if not info.sha:
        raise RuntimeError(
            f"the Hub returned no commit SHA for {model_id!r} revision {requested!r}"
        )
    return str(info.sha)


def broadcast_object(value, rank: int):
    if not dist.is_initialized():
        if value is None:
            raise RuntimeError("rank zero did not provide a value")
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    if values[0] is None:
        raise RuntimeError("rank zero did not provide a value")
    return values[0]


def load_raw_model(
    model_id: str,
    revision: str,
    variant: Variant,
    dtype: torch.dtype,
    device: torch.device,
    tuning: str,
    lora_r: int,
    lora_alpha: int,
    lora_targets: str,
    adapter_init_seed: int,
) -> tuple[torch.nn.Module, dict, dict | None]:
    from transformers import AutoConfig, AutoModelForCausalLM

    validate_kernel_request(
        variant.loss_backend,
        "cross_entropy",
        device,
        dtype,
        tuning=tuning,
        shard="ddp",
    )
    kwargs = attention_load_kwargs(variant.attention_backend, device, dtype)
    if variant.loss_backend == "liger":
        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )
        require_liger_model_support(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=True,
        **kwargs,
    )
    kernel_application = None
    if variant.loss_backend == "liger":
        # Keep this before PEFT: the direct-binding helper accepts only the
        # native Qwen2 class and binds only this instance's forward.
        # Decoder-layer implementations stay identical to reference.
        kernel_application = apply_liger_fused_linear_ce(model)
    resolved_targets = None
    output_head_report = None
    if tuning == "lora":
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(adapter_init_seed)
        torch.cuda.manual_seed_all(adapter_init_seed)
        resolved_targets = resolve_lora_targets(lora_targets, model.config)
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=resolved_targets,
                task_type="CAUSAL_LM",
            ),
        )
    model.to(device)
    model.train()
    model.config.use_cache = False
    resolved_attention_backend(model, variant.attention_backend)
    production_envelope = (
        validate_lora_production_envelope(model)
        if tuning == "lora"
        else None
    )
    if production_envelope is not None:
        output_head_report = production_envelope["output_head"]
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    tuning_report = {
        "mode": tuning,
        "lora_r": lora_r if tuning == "lora" else None,
        "lora_alpha": lora_alpha if tuning == "lora" else None,
        "lora_targets_requested": lora_targets if tuning == "lora" else None,
        "lora_targets_resolved": resolved_targets,
        "adapter_init_seed": adapter_init_seed if tuning == "lora" else None,
        "output_head": output_head_report,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "trainable_dtype_counts": (
            production_envelope["trainable_dtype_counts"]
            if production_envelope is not None
            else {}
        ),
    }
    if production_envelope is None:
        for parameter in model.parameters():
            if parameter.requires_grad:
                dtype_name = str(parameter.dtype).removeprefix("torch.")
                tuning_report["trainable_dtype_counts"][dtype_name] = (
                    tuning_report["trainable_dtype_counts"].get(dtype_name, 0)
                    + parameter.numel()
                )
    return model, tuning_report, kernel_application


def make_batch(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    target_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        generator=generator,
        device=device,
    )
    weights = (
        torch.rand((batch_size, seq_len), generator=generator, device=device)
        < target_fraction
    ).float()
    weights[:, 0] = 0  # token zero has no causal predecessor
    weights[:, -1] = 1  # every rank has a nonempty denominator
    return input_ids, weights


def forward_sum(model, variant: Variant, input_ids, weights):
    if variant.loss_backend == "liger":
        return liger_sft_forward(model, input_ids, weights)
    output = model(input_ids=input_ids, use_cache=False)
    if getattr(output, "logits", None) is None:
        raise RuntimeError("the native benchmark path returned no logits")
    return sft_loss(output.logits, input_ids, weights=weights)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def gradient_snapshot(model) -> dict[str, torch.Tensor]:
    """Copy every trainable gradient to host memory for full parity checks."""
    snapshot: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for name, parameter in unwrap(model).named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        snapshot[name] = parameter.grad.detach().cpu().clone()
    if missing:
        raise RuntimeError(
            "trainable parameters received no gradient during the parity witness: "
            f"{missing[:5]}"
        )
    return snapshot


def trainable_state_digest(model) -> str:
    """Hash trainable names, dtypes, shapes, and exact parameter bytes."""
    digest = hashlib.sha256()
    trainable = 0
    for name, parameter in unwrap(model).named_parameters():
        if not parameter.requires_grad:
            continue
        trainable += 1
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    if not trainable:
        raise RuntimeError("the benchmark model has no trainable parameters")
    return digest.hexdigest()


def snapshot_trainable_state(model) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in unwrap(model).named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise RuntimeError("the benchmark model has no trainable parameters")
    return state


def restore_trainable_state(model, state: dict[str, torch.Tensor]) -> None:
    trainable = {
        name: parameter
        for name, parameter in unwrap(model).named_parameters()
        if parameter.requires_grad
    }
    if trainable.keys() != state.keys():
        missing = sorted(state.keys() - trainable.keys())
        extra = sorted(trainable.keys() - state.keys())
        raise RuntimeError(
            "trainable state layout mismatch while restoring the benchmark anchor: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    with torch.no_grad():
        for name, parameter in trainable.items():
            value = state[name]
            if parameter.shape != value.shape or parameter.dtype != value.dtype:
                raise RuntimeError(
                    f"trainable state metadata mismatch for {name}: "
                    f"parameter={parameter.shape}/{parameter.dtype} "
                    f"anchor={value.shape}/{value.dtype}"
                )
            parameter.copy_(value.to(device=parameter.device))


def lora_factor_nonzero_report(state: dict[str, torch.Tensor]) -> dict:
    report = {}
    for factor in ("lora_A", "lora_B"):
        tensors = [value for name, value in state.items() if factor in name]
        nonzero = sum(int(torch.count_nonzero(value).item()) for value in tensors)
        elements = sum(value.numel() for value in tensors)
        report[factor] = {
            "tensor_count": len(tensors),
            "element_count": elements,
            "nonzero_elements": nonzero,
            "nonzero_fraction": nonzero / elements if elements else None,
        }
        if not tensors or nonzero == 0:
            raise RuntimeError(
                f"controlled native warmup left {factor} absent or entirely zero"
            )
    return report


def tensor_parity(actual, reference, rtol, atol, chunk_size=1_000_000) -> dict:
    """Compare one tensor in bounded chunks and retain scale-aware statistics."""
    if actual.shape != reference.shape:
        return {
            "passed": False,
            "numeric_status": "not_evaluated",
            "shape_matches": False,
            "actual_shape": list(actual.shape),
            "reference_shape": list(reference.shape),
            "element_count": 0,
            "finite_element_count": 0,
            "nonfinite_actual_elements": 0,
            "nonfinite_reference_elements": 0,
            "allclose_violation_count": 0,
            "actual_nonzero_elements": 0,
            "reference_nonzero_elements": 0,
            "max_absolute_error": None,
            "max_relative_error": None,
            "max_actual_absolute": None,
            "max_reference_absolute": None,
            "actual_squared_l2": None,
            "reference_squared_l2": None,
            "difference_squared_l2": None,
            "dot_product": None,
        }

    actual_flat = actual.reshape(-1)
    reference_flat = reference.reshape(-1)
    metrics = {
        "passed": True,
        "numeric_status": "complete",
        "shape_matches": True,
        "actual_shape": list(actual.shape),
        "reference_shape": list(reference.shape),
        "element_count": actual.numel(),
        "finite_element_count": 0,
        "nonfinite_actual_elements": 0,
        "nonfinite_reference_elements": 0,
        "allclose_violation_count": 0,
        "actual_nonzero_elements": 0,
        "reference_nonzero_elements": 0,
        "max_absolute_error": 0.0,
        "max_relative_error": 0.0,
        "max_actual_absolute": 0.0,
        "max_reference_absolute": 0.0,
        "actual_squared_l2": 0.0,
        "reference_squared_l2": 0.0,
        "difference_squared_l2": 0.0,
        "dot_product": 0.0,
    }
    for start in range(0, actual_flat.numel(), chunk_size):
        stop = min(start + chunk_size, actual_flat.numel())
        actual_chunk = actual_flat[start:stop].float()
        reference_chunk = reference_flat[start:stop].float()
        finite_actual = torch.isfinite(actual_chunk)
        finite_reference = torch.isfinite(reference_chunk)
        finite = finite_actual & finite_reference
        nonfinite_actual = int((~finite_actual).sum().item())
        nonfinite_reference = int((~finite_reference).sum().item())
        metrics["nonfinite_actual_elements"] += nonfinite_actual
        metrics["nonfinite_reference_elements"] += nonfinite_reference
        metrics["finite_element_count"] += int(finite.sum().item())
        metrics["actual_nonzero_elements"] += int(
            ((actual_chunk != 0) & finite).sum().item()
        )
        metrics["reference_nonzero_elements"] += int(
            ((reference_chunk != 0) & finite).sum().item()
        )

        violations = ~finite
        if bool(finite.any().item()):
            finite_actual_chunk = actual_chunk[finite]
            finite_reference_chunk = reference_chunk[finite]
            actual_double = finite_actual_chunk.double()
            reference_double = finite_reference_chunk.double()
            difference_double = actual_double - reference_double
            difference = difference_double.abs()
            relative = difference / reference_double.abs().clamp_min(1e-12)
            finite_violations = difference > (
                atol + rtol * reference_double.abs()
            )
            violations = violations.clone()
            violations[finite] = finite_violations

            metrics["max_absolute_error"] = max(
                metrics["max_absolute_error"], float(difference.max().item())
            )
            metrics["max_relative_error"] = max(
                metrics["max_relative_error"], float(relative.max().item())
            )
            metrics["max_actual_absolute"] = max(
                metrics["max_actual_absolute"],
                float(finite_actual_chunk.abs().max().item()),
            )
            metrics["max_reference_absolute"] = max(
                metrics["max_reference_absolute"],
                float(finite_reference_chunk.abs().max().item()),
            )
            metrics["actual_squared_l2"] += float(
                actual_double.square().sum().item()
            )
            metrics["reference_squared_l2"] += float(
                reference_double.square().sum().item()
            )
            metrics["difference_squared_l2"] += float(
                difference_double.square().sum().item()
            )
            metrics["dot_product"] += float(
                (actual_double * reference_double).sum().item()
            )
        metrics["allclose_violation_count"] += int(violations.sum().item())

    if metrics["nonfinite_actual_elements"] or metrics["nonfinite_reference_elements"]:
        metrics["numeric_status"] = "partial"
    if metrics["finite_element_count"] == 0:
        metrics["max_absolute_error"] = None
        metrics["max_relative_error"] = None
        metrics["max_actual_absolute"] = None
        metrics["max_reference_absolute"] = None
        metrics["actual_squared_l2"] = None
        metrics["reference_squared_l2"] = None
        metrics["difference_squared_l2"] = None
        metrics["dot_product"] = None
    metrics["passed"] = metrics["allclose_violation_count"] == 0
    return metrics


def _empty_map_parity() -> dict:
    return {
        "status": "passed",
        "structural_status": "passed",
        "finiteness_status": "passed",
        "numeric_status": "complete",
        "numeric_scope": "compatible_finite_elements",
        "checked_tensors": 0,
        "expected_tensors": 0,
        "element_count": 0,
        "finite_element_count": 0,
        "nonfinite_actual_elements": 0,
        "nonfinite_reference_elements": 0,
        "allclose_violation_count": 0,
        "actual_nonzero_elements": 0,
        "reference_nonzero_elements": 0,
        "max_absolute_error": 0.0,
        "max_relative_error": 0.0,
        "max_actual_absolute": 0.0,
        "max_reference_absolute": 0.0,
        "actual_squared_l2": 0.0,
        "reference_squared_l2": 0.0,
        "difference_squared_l2": 0.0,
        "dot_product": 0.0,
        "first_failing_tensor": None,
        "worst_failing_tensor": None,
        "worst_failing_tensor_max_absolute_error": None,
        "worst_failing_tensor_numeric_status": None,
        "_worst_failure_sort_key": None,
        "missing_tensors": [],
        "extra_tensors": [],
        "shape_mismatches": [],
        "reasons": [],
    }


def _finalize_map_parity(summary: dict) -> dict:
    summary.pop("_worst_failure_sort_key")
    actual_squared_l2 = summary.pop("actual_squared_l2")
    reference_squared_l2 = summary.pop("reference_squared_l2")
    difference_squared_l2 = summary.pop("difference_squared_l2")
    dot_product = summary.pop("dot_product")
    numeric_evaluable = summary["finite_element_count"] > 0
    if not numeric_evaluable:
        summary["numeric_status"] = "not_evaluated"
    raw_moments = (
        actual_squared_l2,
        reference_squared_l2,
        difference_squared_l2,
        dot_product,
    )
    if numeric_evaluable and not all(math.isfinite(value) for value in raw_moments):
        summary["numeric_status"] = "not_evaluated"
        numeric_evaluable = False
    if not numeric_evaluable:
        summary.update(
            max_absolute_error=None,
            max_relative_error=None,
            max_actual_absolute=None,
            max_reference_absolute=None,
            actual_l2_norm=None,
            reference_l2_norm=None,
            difference_l2_norm=None,
            relative_l2_error=None,
            cosine_similarity=None,
            allclose_violation_fraction=(
                summary["allclose_violation_count"] / summary["element_count"]
                if summary["element_count"]
                else None
            ),
            actual_nonzero_fraction=(
                summary["actual_nonzero_elements"]
                / summary["finite_element_count"]
                if summary["finite_element_count"]
                else None
            ),
            reference_nonzero_fraction=(
                summary["reference_nonzero_elements"]
                / summary["finite_element_count"]
                if summary["finite_element_count"]
                else None
            ),
            numeric_element_count=summary["finite_element_count"],
            numeric_element_fraction=(
                summary["finite_element_count"] / summary["element_count"]
                if summary["element_count"]
                else None
            ),
        )
        return summary

    actual_norm = math.sqrt(max(0.0, actual_squared_l2))
    reference_norm = math.sqrt(max(0.0, reference_squared_l2))
    difference_norm = math.sqrt(max(0.0, difference_squared_l2))
    denominator = actual_norm * reference_norm
    if denominator:
        cosine = max(-1.0, min(1.0, dot_product / denominator))
    elif difference_norm == 0:
        cosine = 1.0
    else:
        cosine = None
    summary.update(
        actual_l2_norm=actual_norm,
        reference_l2_norm=reference_norm,
        difference_l2_norm=difference_norm,
        relative_l2_error=difference_norm / max(reference_norm, 1e-12),
        cosine_similarity=cosine,
        allclose_violation_fraction=(
            summary["allclose_violation_count"] / summary["element_count"]
        ),
        actual_nonzero_fraction=(
            summary["actual_nonzero_elements"] / summary["finite_element_count"]
        ),
        reference_nonzero_fraction=(
            summary["reference_nonzero_elements"] / summary["finite_element_count"]
        ),
        numeric_element_count=summary["finite_element_count"],
        numeric_element_fraction=(
            summary["finite_element_count"] / summary["element_count"]
        ),
    )
    return summary


def compare_tensor_maps(actual: dict, reference: dict, rtol: float, atol: float) -> dict:
    """Compare every common tensor and report structural and numeric failures."""
    summary = _empty_map_parity()
    summary["expected_tensors"] = len(reference)
    summary["missing_tensors"] = sorted(reference.keys() - actual.keys())
    summary["extra_tensors"] = sorted(actual.keys() - reference.keys())
    if summary["missing_tensors"] or summary["extra_tensors"]:
        summary["status"] = "failed"
        summary["structural_status"] = "failed"
        summary["numeric_status"] = "partial"
        summary["reasons"].append(
            "key mismatch: "
            f"missing={summary['missing_tensors'][:5]} "
            f"extra={summary['extra_tensors'][:5]}"
        )

    for name, reference_tensor in reference.items():
        if name not in actual:
            continue
        metrics = tensor_parity(actual[name], reference_tensor, rtol, atol)
        summary["checked_tensors"] += 1
        if metrics["numeric_status"] != "complete":
            summary["numeric_status"] = "partial"
        if not metrics["shape_matches"]:
            summary["shape_mismatches"].append(
                {
                    "tensor": name,
                    "actual": metrics["actual_shape"],
                    "reference": metrics["reference_shape"],
                }
            )
        for key in (
            "element_count",
            "finite_element_count",
            "nonfinite_actual_elements",
            "nonfinite_reference_elements",
            "allclose_violation_count",
            "actual_nonzero_elements",
            "reference_nonzero_elements",
            "actual_squared_l2",
            "reference_squared_l2",
            "difference_squared_l2",
            "dot_product",
        ):
            if metrics[key] is not None:
                summary[key] += metrics[key]
        for key in (
            "max_absolute_error",
            "max_relative_error",
            "max_actual_absolute",
            "max_reference_absolute",
        ):
            value = metrics[key]
            if value is not None:
                summary[key] = max(summary[key], value)
        if not metrics["passed"]:
            summary["status"] = "failed"
            if summary["first_failing_tensor"] is None:
                summary["first_failing_tensor"] = name
            failure_priority = {
                "complete": 1,
                "partial": 2,
                "not_evaluated": 2,
            }[metrics["numeric_status"]]
            failure_sort_key = (
                failure_priority,
                metrics["max_absolute_error"] or 0.0,
            )
            if (
                summary["_worst_failure_sort_key"] is None
                or failure_sort_key > summary["_worst_failure_sort_key"]
            ):
                summary["_worst_failure_sort_key"] = failure_sort_key
                summary["worst_failing_tensor"] = name
                summary["worst_failing_tensor_max_absolute_error"] = metrics[
                    "max_absolute_error"
                ]
                summary["worst_failing_tensor_numeric_status"] = metrics[
                    "numeric_status"
                ]

    if summary["shape_mismatches"]:
        summary["status"] = "failed"
        summary["structural_status"] = "failed"
        summary["numeric_status"] = "partial"
        summary["reasons"].append(
            f"shape mismatch for {summary['shape_mismatches'][0]['tensor']}"
        )
    if summary["nonfinite_actual_elements"] or summary["nonfinite_reference_elements"]:
        summary["finiteness_status"] = "failed"
        summary["numeric_status"] = "partial"
        summary["reasons"].append(
            "nonfinite values: "
            f"actual={summary['nonfinite_actual_elements']} "
            f"reference={summary['nonfinite_reference_elements']}"
        )
    if summary["first_failing_tensor"] is not None:
        summary["reasons"].append(
            f"numeric parity failed for {summary['first_failing_tensor']}"
        )
    if not reference and not actual:
        summary["status"] = "not_evaluated"
        summary["numeric_status"] = "not_evaluated"
        summary["reasons"].append("no tensors were available for parity")
    return _finalize_map_parity(summary)


def _update_sensitivity(summary: dict, side: str) -> dict:
    nonzero_elements = summary[f"{side}_nonzero_elements"]
    element_count = summary["element_count"]
    nonfinite_elements = summary[f"nonfinite_{side}_elements"]
    max_absolute = summary[f"max_{side}_absolute"]
    l2_norm = summary[f"{side}_l2_norm"]
    if element_count == 0:
        status = "empty"
    elif nonfinite_elements:
        status = "nonfinite"
    elif nonzero_elements == 0 or max_absolute == 0 or l2_norm == 0:
        status = "rounded_away"
    else:
        status = "meaningful"
    return {
        "status": status,
        "meaningful": status == "meaningful",
        "element_count": element_count,
        "nonzero_elements": nonzero_elements,
        "nonzero_fraction": (
            nonzero_elements / element_count if element_count else 0.0
        ),
        "nonfinite_elements": nonfinite_elements,
        "max_absolute": max_absolute,
        "l2_norm": l2_norm,
    }


def compare_parity(
    loss: float,
    gradients: dict[str, torch.Tensor],
    reference_loss: float,
    reference_gradients: dict[str, torch.Tensor],
    rtol: float,
    atol: float,
    parameter_deltas: dict[str, torch.Tensor] | None = None,
    reference_parameter_deltas: dict[str, torch.Tensor] | None = None,
    parameter_delta_rtol: float | None = None,
    parameter_delta_atol: float | None = None,
) -> dict:
    loss_is_finite = math.isfinite(loss)
    reference_loss_is_finite = math.isfinite(reference_loss)
    losses_are_finite = loss_is_finite and reference_loss_is_finite
    raw_loss_abs_error = abs(loss - reference_loss) if losses_are_finite else None
    loss_error_is_finite = raw_loss_abs_error is not None and math.isfinite(
        raw_loss_abs_error
    )
    loss_passed = losses_are_finite and math.isclose(
        loss, reference_loss, rel_tol=rtol, abs_tol=atol
    )
    if not losses_are_finite:
        loss_status = "nonfinite"
    elif not loss_error_is_finite:
        loss_status = "overflow"
        loss_passed = False
    else:
        loss_status = "passed" if loss_passed else "failed"
    gradient_parity = compare_tensor_maps(gradients, reference_gradients, rtol, atol)
    result = {
        "passed": False,
        "loss": loss if loss_is_finite else None,
        "reference_loss": reference_loss if reference_loss_is_finite else None,
        "loss_abs_error": raw_loss_abs_error if loss_error_is_finite else None,
        "loss_status": loss_status,
        "loss_nonfinite_actual": not loss_is_finite,
        "loss_nonfinite_reference": not reference_loss_is_finite,
        "gradient_status": gradient_parity["status"],
        "gradient_parity": gradient_parity,
        "max_gradient_abs_error": gradient_parity["max_absolute_error"],
        "max_gradient_relative_error": gradient_parity["max_relative_error"],
        "gradient_relative_l2_error": gradient_parity["relative_l2_error"],
        "gradient_cosine_similarity": gradient_parity["cosine_similarity"],
        "gradient_allclose_violation_fraction": gradient_parity[
            "allclose_violation_fraction"
        ],
        "checked_gradient_tensors": gradient_parity["checked_tensors"],
        "first_failing_gradient_tensor": gradient_parity[
            "first_failing_tensor"
        ],
        "worst_failing_gradient_tensor": gradient_parity[
            "worst_failing_tensor"
        ],
        "parameter_delta_status": "not_evaluated",
        "parameter_delta_parity": None,
        "parameter_delta_actual_sensitivity": None,
        "parameter_delta_reference_sensitivity": None,
        "max_parameter_delta_abs_error": None,
        "max_parameter_delta_relative_error": None,
        "checked_parameter_delta_tensors": 0,
        "first_failing_parameter_delta_tensor": None,
        "worst_failing_parameter_delta_tensor": None,
        "reasons": [],
        "reason": None,
    }
    if not loss_passed:
        result["reasons"].append(
            "loss parity gate failed"
            if losses_are_finite and loss_error_is_finite
            else (
                "loss parity gate received nonfinite values"
                if not losses_are_finite
                else "loss parity error overflowed"
            )
        )
    result["reasons"].extend(
        f"gradient {reason}" for reason in gradient_parity["reasons"]
    )

    incomplete_deltas = (parameter_deltas is None) != (
        reference_parameter_deltas is None
    )
    if incomplete_deltas:
        result["parameter_delta_status"] = "failed"
        result["reasons"].append("parameter-delta parity inputs are incomplete")
    elif parameter_deltas is not None:
        delta_rtol = rtol if parameter_delta_rtol is None else parameter_delta_rtol
        delta_atol = atol if parameter_delta_atol is None else parameter_delta_atol
        delta_parity = compare_tensor_maps(
            parameter_deltas,
            reference_parameter_deltas,
            delta_rtol,
            delta_atol,
        )
        actual_sensitivity = _update_sensitivity(delta_parity, "actual")
        reference_sensitivity = _update_sensitivity(delta_parity, "reference")
        delta_status = delta_parity["status"]
        if delta_status == "passed" and not (
            actual_sensitivity["meaningful"]
            and reference_sensitivity["meaningful"]
        ):
            delta_status = "not_meaningful"
            result["reasons"].append(
                "parameter-delta witness is insensitive because updates were rounded away"
            )
        result.update(
            parameter_delta_status=delta_status,
            parameter_delta_parity=delta_parity,
            parameter_delta_actual_sensitivity=actual_sensitivity,
            parameter_delta_reference_sensitivity=reference_sensitivity,
            max_parameter_delta_abs_error=delta_parity["max_absolute_error"],
            max_parameter_delta_relative_error=delta_parity[
                "max_relative_error"
            ],
            checked_parameter_delta_tensors=delta_parity["checked_tensors"],
            first_failing_parameter_delta_tensor=delta_parity[
                "first_failing_tensor"
            ],
            worst_failing_parameter_delta_tensor=delta_parity[
                "worst_failing_tensor"
            ],
        )
        result["reasons"].extend(
            f"parameter-delta {reason}" for reason in delta_parity["reasons"]
        )

    delta_passed = result["parameter_delta_status"] in ("passed", "not_evaluated")
    result["passed"] = (
        loss_passed and gradient_parity["status"] == "passed" and delta_passed
    )
    result["reason"] = result["reasons"][0] if result["reasons"] else None
    return result


def compact_parity_diagnostic(parity: dict, rank: int) -> dict:
    """Extract the bounded per-rank evidence needed for distributed reporting."""
    gradient = parity["gradient_parity"]
    delta = parity.get("parameter_delta_parity")
    return {
        "rank": rank,
        "passed": bool(parity["passed"]),
        "reason": parity.get("reason"),
        "loss_status": parity["loss_status"],
        "loss_abs_error": parity["loss_abs_error"],
        "loss_nonfinite_actual": parity["loss_nonfinite_actual"],
        "loss_nonfinite_reference": parity["loss_nonfinite_reference"],
        "gradient_status": parity["gradient_status"],
        "gradient_max_absolute_error": parity["max_gradient_abs_error"],
        "gradient_max_relative_error": parity["max_gradient_relative_error"],
        "gradient_relative_l2_error": parity["gradient_relative_l2_error"],
        "gradient_cosine_similarity": parity["gradient_cosine_similarity"],
        "gradient_allclose_violation_fraction": parity[
            "gradient_allclose_violation_fraction"
        ],
        "gradient_nonfinite_actual_elements": gradient[
            "nonfinite_actual_elements"
        ],
        "gradient_nonfinite_reference_elements": gradient[
            "nonfinite_reference_elements"
        ],
        "first_failing_gradient_tensor": parity[
            "first_failing_gradient_tensor"
        ],
        "worst_failing_gradient_tensor": parity[
            "worst_failing_gradient_tensor"
        ],
        "parameter_delta_status": parity["parameter_delta_status"],
        "parameter_delta_max_absolute_error": parity[
            "max_parameter_delta_abs_error"
        ],
        "parameter_delta_max_relative_error": parity[
            "max_parameter_delta_relative_error"
        ],
        "parameter_delta_relative_l2_error": (
            delta["relative_l2_error"] if delta is not None else None
        ),
        "parameter_delta_allclose_violation_fraction": (
            delta["allclose_violation_fraction"] if delta is not None else None
        ),
        "parameter_delta_nonfinite_actual_elements": (
            delta["nonfinite_actual_elements"] if delta is not None else 0
        ),
        "parameter_delta_nonfinite_reference_elements": (
            delta["nonfinite_reference_elements"] if delta is not None else 0
        ),
        "first_failing_parameter_delta_tensor": parity[
            "first_failing_parameter_delta_tensor"
        ],
        "worst_failing_parameter_delta_tensor": parity[
            "worst_failing_parameter_delta_tensor"
        ],
    }


def _defined_metric(diagnostics: list[dict], key: str, reducer):
    values = [item[key] for item in diagnostics if item[key] is not None]
    return reducer(values) if values else None


def _aggregate_status(diagnostics: list[dict], key: str) -> str:
    statuses = [item[key] for item in diagnostics]
    for status in (
        "nonfinite",
        "overflow",
        "failed",
        "not_meaningful",
        "not_evaluated",
    ):
        if status in statuses:
            return status
    return "passed"


def aggregate_parity_diagnostics(diagnostics: list[dict]) -> dict:
    """Produce strict-JSON global parity evidence from all rank summaries."""
    if not diagnostics:
        raise ValueError("at least one parity diagnostic is required")
    ordered = sorted(diagnostics, key=lambda item: item["rank"])
    ranks = [item["rank"] for item in ordered]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate parity diagnostic ranks: {ranks}")
    failing = [item for item in ordered if not item["passed"]]

    def failure_score(item: dict) -> tuple:
        nonfinite_loss = (
            item["loss_nonfinite_actual"] or item["loss_nonfinite_reference"]
        )
        nonfinite = sum(
            item[key]
            for key in (
                "gradient_nonfinite_actual_elements",
                "gradient_nonfinite_reference_elements",
                "parameter_delta_nonfinite_actual_elements",
                "parameter_delta_nonfinite_reference_elements",
            )
        )
        failed_components = sum(
            item[key] != "passed"
            for key in ("loss_status", "gradient_status", "parameter_delta_status")
        )
        numeric_severity = max(
            value
            for value in (
                item["gradient_relative_l2_error"],
                item["gradient_allclose_violation_fraction"],
                item["parameter_delta_relative_l2_error"],
                item["parameter_delta_allclose_violation_fraction"],
                item["loss_abs_error"],
                0.0,
            )
            if value is not None
        )
        return (
            nonfinite_loss,
            nonfinite > 0,
            failed_components,
            numeric_severity,
            -item["rank"],
        )

    worst = max(failing, key=failure_score) if failing else None
    return {
        "passed": not failing,
        "rank_count": len(ordered),
        "ranks": ranks,
        "failing_ranks": [item["rank"] for item in failing],
        "failing_rank_diagnostics": failing,
        "worst_failing_rank": worst["rank"] if worst else None,
        "worst_failure_reason": worst["reason"] if worst else None,
        "loss_status": _aggregate_status(ordered, "loss_status"),
        "loss_abs_error_max": _defined_metric(ordered, "loss_abs_error", max),
        "nonfinite_loss_ranks": [
            item["rank"]
            for item in ordered
            if item["loss_nonfinite_actual"]
            or item["loss_nonfinite_reference"]
        ],
        "loss_nonfinite_actual_ranks": [
            item["rank"] for item in ordered if item["loss_nonfinite_actual"]
        ],
        "loss_nonfinite_reference_ranks": [
            item["rank"] for item in ordered if item["loss_nonfinite_reference"]
        ],
        "gradient_status": _aggregate_status(ordered, "gradient_status"),
        "gradient_max_absolute_error_max": _defined_metric(
            ordered, "gradient_max_absolute_error", max
        ),
        "gradient_max_relative_error_max": _defined_metric(
            ordered, "gradient_max_relative_error", max
        ),
        "gradient_relative_l2_error_max": _defined_metric(
            ordered, "gradient_relative_l2_error", max
        ),
        "gradient_cosine_similarity_min": _defined_metric(
            ordered, "gradient_cosine_similarity", min
        ),
        "gradient_allclose_violation_fraction_max": _defined_metric(
            ordered, "gradient_allclose_violation_fraction", max
        ),
        "gradient_nonfinite_actual_elements_total": sum(
            item["gradient_nonfinite_actual_elements"] for item in ordered
        ),
        "gradient_nonfinite_reference_elements_total": sum(
            item["gradient_nonfinite_reference_elements"] for item in ordered
        ),
        "parameter_delta_status": _aggregate_status(
            ordered, "parameter_delta_status"
        ),
        "parameter_delta_max_absolute_error_max": _defined_metric(
            ordered, "parameter_delta_max_absolute_error", max
        ),
        "parameter_delta_max_relative_error_max": _defined_metric(
            ordered, "parameter_delta_max_relative_error", max
        ),
        "parameter_delta_relative_l2_error_max": _defined_metric(
            ordered, "parameter_delta_relative_l2_error", max
        ),
        "parameter_delta_allclose_violation_fraction_max": _defined_metric(
            ordered, "parameter_delta_allclose_violation_fraction", max
        ),
        "parameter_delta_nonfinite_actual_elements_total": sum(
            item["parameter_delta_nonfinite_actual_elements"] for item in ordered
        ),
        "parameter_delta_nonfinite_reference_elements_total": sum(
            item["parameter_delta_nonfinite_reference_elements"] for item in ordered
        ),
    }


def gather_parity_diagnostics(parity: dict, rank: int, world: int) -> list[dict]:
    local = compact_parity_diagnostic(parity, rank)
    if not dist.is_initialized():
        return [local]
    diagnostics: list[dict | None] = [None] * world
    dist.all_gather_object(diagnostics, local)
    if any(item is None for item in diagnostics):
        raise RuntimeError("distributed parity diagnostic gather was incomplete")
    return [item for item in diagnostics if item is not None]


def aggregate_state_digest_diagnostics(
    diagnostics: list[dict], reference_digest: str | None = None
) -> dict:
    """Build one consistent all-rank schema for trainable-state identity."""
    if not diagnostics:
        raise ValueError("at least one state-digest diagnostic is required")
    ordered = sorted(diagnostics, key=lambda item: item["rank"])
    ranks = [item["rank"] for item in ordered]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate state-digest diagnostic ranks: {ranks}")
    if reference_digest is None:
        rank_zero = next((item for item in ordered if item["rank"] == 0), None)
        if rank_zero is None:
            raise ValueError("rank 0 is required to establish a state-digest reference")
        reference_digest = rank_zero["digest"]
    rank_digests = [
        {
            "rank": item["rank"],
            "digest": item["digest"],
            "matches_reference": item["digest"] == reference_digest,
        }
        for item in ordered
    ]
    failing_ranks = [
        item["rank"] for item in rank_digests if not item["matches_reference"]
    ]
    return {
        "passed": not failing_ranks,
        "reference_digest": reference_digest,
        "rank_digests": rank_digests,
        "failing_ranks": failing_ranks,
        "unique_digest_count": len({item["digest"] for item in rank_digests}),
    }


def gather_state_digest_diagnostics(
    digest: str,
    rank: int,
    world: int,
    reference_digest: str | None = None,
) -> dict:
    local = {"rank": rank, "digest": digest}
    if not dist.is_initialized():
        diagnostics = [local]
    else:
        gathered: list[dict | None] = [None] * world
        dist.all_gather_object(gathered, local)
        if any(item is None for item in gathered):
            raise RuntimeError("distributed state-digest gather was incomplete")
        diagnostics = [item for item in gathered if item is not None]
    return aggregate_state_digest_diagnostics(diagnostics, reference_digest)


def apply_distributed_parity(parity: dict, diagnostics: list[dict], rank: int) -> dict:
    """Promote global rank evidence into the top-level parity decision."""
    distributed = aggregate_parity_diagnostics(diagnostics)
    parity["detailed_metrics_scope"] = "local_rank"
    parity["detailed_metrics_rank"] = rank
    parity["rank_diagnostics"] = diagnostics
    parity["distributed_parity"] = distributed
    parity["passed_local_rank"] = parity["passed"]
    parity["passed"] = distributed["passed"]
    parity["passed_all_ranks"] = distributed["passed"]
    parity["loss_status"] = distributed["loss_status"]
    parity["gradient_status"] = distributed["gradient_status"]
    parity["parameter_delta_status"] = distributed["parameter_delta_status"]
    parity["loss_abs_error"] = distributed["loss_abs_error_max"]
    parity["loss_nonfinite_actual"] = bool(
        distributed["loss_nonfinite_actual_ranks"]
    )
    parity["loss_nonfinite_reference"] = bool(
        distributed["loss_nonfinite_reference_ranks"]
    )
    parity["nonfinite_loss_ranks"] = distributed["nonfinite_loss_ranks"]
    parity["max_gradient_abs_error"] = distributed[
        "gradient_max_absolute_error_max"
    ]
    parity["max_gradient_relative_error"] = distributed[
        "gradient_max_relative_error_max"
    ]
    parity["gradient_relative_l2_error"] = distributed[
        "gradient_relative_l2_error_max"
    ]
    parity["gradient_cosine_similarity"] = distributed[
        "gradient_cosine_similarity_min"
    ]
    parity["gradient_allclose_violation_fraction"] = distributed[
        "gradient_allclose_violation_fraction_max"
    ]
    parity["max_parameter_delta_abs_error"] = distributed[
        "parameter_delta_max_absolute_error_max"
    ]
    parity["max_parameter_delta_relative_error"] = distributed[
        "parameter_delta_max_relative_error_max"
    ]
    if not distributed["passed"]:
        parity["reason"] = distributed["worst_failure_reason"]
    return parity


def backward_and_clip(
    model,
    loss: torch.Tensor,
    local_target_tokens: torch.Tensor,
    tuning: str,
    world: int,
    device: torch.device,
) -> dict:
    """Apply the exact global-token objective and production gradient policy."""
    global_target_tokens = local_target_tokens.detach().to(
        device=device, dtype=torch.long
    )
    if world > 1:
        if not dist.is_initialized():
            raise RuntimeError("world > 1 requires initialized distributed state")
        dist.all_reduce(global_target_tokens, op=dist.ReduceOp.SUM)
    denominator = int(global_target_tokens.item())
    if denominator < 1:
        raise RuntimeError("the parity batch contains no positive target tokens")

    # DDP and the manual LoRA reduction both average gradients across ranks.
    # Scaling every local token-sum loss by world/global_targets therefore
    # yields the exact island-global token-mean gradient after that average.
    (loss * (world / denominator)).backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if tuning == "lora":
        allreduce_trainable_grads(trainable, world)
    pre_clip_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    pre_clip_norm_value = float(pre_clip_norm.detach().float().item())
    return {
        "global_target_tokens": denominator,
        "pre_clip_grad_norm": (
            pre_clip_norm_value if math.isfinite(pre_clip_norm_value) else None
        ),
        "pre_clip_grad_norm_nonfinite": not math.isfinite(pre_clip_norm_value),
        "clip_max_norm": 1.0,
    }


def parity_witness(model, variant, input_ids, weights, tuning, world, device):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss, target_tokens = forward_sum(model, variant, input_ids, weights)
    backward_report = backward_and_clip(
        model, loss, target_tokens, tuning, world, device
    )
    torch.cuda.synchronize(device)
    compile_seconds = time.perf_counter() - started
    witness = gradient_snapshot(model)
    value = float(loss.detach().float().item())
    return value, witness, compile_seconds, backward_report


def parameter_delta_witness(model, optimizer, device, restore_parameters=False):
    """Apply one optimizer step and retain every trainable parameter delta.

    ``restore_parameters`` is used only by the reference-repeat control. It
    restores the exact pre-step values after observing the update so a fresh
    optimizer can repeat the witness from identical state.
    """
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in unwrap(model).named_parameters()
        if parameter.requires_grad
    }
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize(device)
    optimizer_step_seconds = time.perf_counter() - started
    deltas = {}
    for name, parameter in unwrap(model).named_parameters():
        if not parameter.requires_grad:
            continue
        after = parameter.detach().cpu()
        initial = before.pop(name)
        deltas[name] = after.float() - initial.float()
        if restore_parameters:
            with torch.no_grad():
                parameter.copy_(initial.to(device=parameter.device))
    if before:
        raise RuntimeError(
            "trainable parameters disappeared during optimizer step: "
            f"{list(before)[:5]}"
        )
    if restore_parameters:
        torch.cuda.synchronize(device)
    optimizer.zero_grad(set_to_none=True)
    return deltas, optimizer_step_seconds


def make_optimizer(model, learning_rate: float, weight_decay: float):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("the benchmark model has no trainable parameters")
    return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)


def training_step(
    model, optimizer, variant, input_ids, weights, tuning, world, device
) -> dict:
    optimizer.zero_grad(set_to_none=True)
    loss, target_tokens = forward_sum(model, variant, input_ids, weights)
    backward_report = backward_and_clip(
        model, loss, target_tokens, tuning, world, device
    )
    optimizer.step()
    return backward_report


def benchmark_variant(
    model,
    optimizer,
    variant,
    input_ids,
    weights,
    warmup_steps: int,
    measured_steps: int,
    tuning: str,
    world: int,
    device: torch.device,
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    training_step(
        model, optimizer, variant, input_ids, weights, tuning, world, device
    )
    torch.cuda.synchronize(device)
    first_optimizer_step_seconds = distributed_max(time.perf_counter() - started, device)

    for _ in range(warmup_steps):
        training_step(
            model, optimizer, variant, input_ids, weights, tuning, world, device
        )
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()

    durations: list[float] = []
    for _ in range(measured_steps):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        training_step(
            model, optimizer, variant, input_ids, weights, tuning, world, device
        )
        torch.cuda.synchronize(device)
        durations.append(distributed_max(time.perf_counter() - started, device))

    raw_tokens_per_step = distributed_sum(input_ids.numel(), device)
    local_target_tokens = int((weights[:, 1:] > 0).sum().item())
    target_tokens_per_step = distributed_sum(local_target_tokens, device)
    measured_seconds = sum(durations)
    peak_allocated = distributed_max(torch.cuda.max_memory_allocated(device), device)
    peak_reserved = distributed_max(torch.cuda.max_memory_reserved(device), device)
    return {
        "world_size": world,
        "steps": measured_steps,
        "warmup_steps": warmup_steps,
        "first_optimizer_step_seconds": first_optimizer_step_seconds,
        "p50_step_seconds": percentile(durations, 0.50),
        "p95_step_seconds": percentile(durations, 0.95),
        "mean_step_seconds": statistics.fmean(durations),
        "raw_tokens_per_step": raw_tokens_per_step,
        "target_tokens_per_step": target_tokens_per_step,
        "raw_tokens_per_second": raw_tokens_per_step * measured_steps / measured_seconds,
        "target_tokens_per_second": target_tokens_per_step
        * measured_steps
        / measured_seconds,
        "peak_allocated_bytes_per_gpu_max": int(peak_allocated),
        "peak_reserved_bytes_per_gpu_max": int(peak_reserved),
    }


def cleanup_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="public Hugging Face model ID or an alias resolving to one",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hub branch, tag, or commit (default main); always resolved to and loaded by exact SHA",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help=(
            "all or comma-separated component-isolated variants: "
            + ", ".join(VARIANTS_BY_NAME)
        ),
    )
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--tuning",
        choices=["lora", "full"],
        default="lora",
        help="train FP32 LoRA adapters by default; full is a native-only separate profile",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--lora-targets",
        choices=["auto", "attention", "all-linear"],
        default="auto",
    )
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--parity-micro-batch-size", type=int, default=1)
    parser.add_argument("--parity-seq-len", type=int, default=128)
    parser.add_argument("--target-fraction", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--trial-index",
        type=int,
        default=1,
        help="independent invocation index written to JSON for external aggregation",
    )
    parser.add_argument("--parity-rtol", type=float, default=5e-2)
    parser.add_argument("--parity-atol", type=float, default=5e-3)
    parser.add_argument("--parameter-delta-rtol", type=float, default=5e-2)
    parser.add_argument("--parameter-delta-atol", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, default=Path("a100-kernel-benchmark.json"))
    return parser


def validate_args(args) -> None:
    if args.micro_batch_size < 1 or args.parity_micro_batch_size < 1:
        raise ValueError("micro-batch sizes must be positive")
    if args.seq_len < 2 or args.parity_seq_len < 2:
        raise ValueError("sequence lengths must be at least two")
    if args.steps < 1 or args.warmup_steps < 0:
        raise ValueError("--steps must be positive and --warmup-steps nonnegative")
    if not 0 < args.target_fraction <= 1:
        raise ValueError("--target-fraction must be in (0, 1]")
    if args.trial_index < 1:
        raise ValueError("--trial-index must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("--learning-rate must be positive and --weight-decay nonnegative")
    if args.lora_r < 1 or args.lora_alpha < 1:
        raise ValueError("--lora-r and --lora-alpha must be positive")
    if min(
        args.parity_rtol,
        args.parity_atol,
        args.parameter_delta_rtol,
        args.parameter_delta_atol,
    ) < 0:
        raise ValueError("parity tolerances must be nonnegative")
    selected = select_variants(args.variants)
    if args.tuning != "lora" and any(
        variant.loss_backend == "liger" for variant in selected
    ):
        raise ValueError(
            "the fused-linear-CE benchmark arm is approved only for --tuning lora; "
            "select native-only variants for a separate full-tuning profile"
        )


def _validated_git_sha(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(
            "YETO_GIT_SHA must be a 40- or 64-character hexadecimal git object ID"
        )
    return normalized


def _validated_dirty(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true"):
        return True
    if normalized in ("0", "false"):
        return False
    raise ValueError("YETO_GIT_DIRTY must be one of true, false, 1, or 0")


def _git_output(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def source_provenance() -> dict:
    """Resolve strict source identity from overrides or the local git worktree."""
    override_sha = os.environ.get("YETO_GIT_SHA")
    override_dirty = os.environ.get("YETO_GIT_DIRTY")
    if (override_sha is None) != (override_dirty is None):
        raise ValueError(
            "YETO_GIT_SHA and YETO_GIT_DIRTY must be supplied together"
        )
    if override_sha is not None:
        git_sha = _validated_git_sha(override_sha)
        git_dirty = _validated_dirty(override_dirty)
        provenance_source = "environment_override"
    else:
        raw_sha = _git_output(["rev-parse", "HEAD"])
        raw_status = _git_output(["status", "--porcelain", "--untracked-files=normal"])
        if raw_sha is None or raw_status is None:
            raise RuntimeError(
                "source provenance is unavailable; in a synchronized workdir "
                "without .git, set both YETO_GIT_SHA and YETO_GIT_DIRTY"
            )
        git_sha = _validated_git_sha(raw_sha)
        git_dirty = bool(raw_status)
        provenance_source = "git_worktree"

    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "benchmark_script_sha256": script_sha256,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "provenance_source": provenance_source,
        "clean_commit_exact": not git_dirty,
    }


def environment_report(args, world, device) -> dict:
    capability = torch.cuda.get_device_capability(device)
    provenance = source_provenance()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "accelerate": package_version("accelerate"),
        "source_provenance": provenance,
        "benchmark_script_sha256": provenance["benchmark_script_sha256"],
        "yeto_git_sha": provenance["git_sha"],
        "yeto_git_dirty": provenance["git_dirty"],
        "provenance_source": provenance["provenance_source"],
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "world_size": world,
        "liger_kernel": package_version("liger-kernel"),
        "flash_attn": package_version("flash-attn"),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def finalize_report_status(
    report: dict,
    failed: bool,
    fatal_phase: str | None,
    fatal_reason: str | None,
) -> None:
    report["completed_variants"] = [
        record["variant"]["name"] for record in report["variants"]
    ]
    if failed:
        report["status"] = "failed"
        report["fatal"] = {
            "phase": fatal_phase or "unknown",
            "reason": fatal_reason or "benchmark failed without a recorded reason",
        }
    elif (
        len(report["completed_variants"]) == len(report["planned_variants"])
        and all(record["status"] == "passed" for record in report["variants"])
    ):
        report["status"] = "passed"
        report["fatal"] = None
    else:
        report["status"] = "incomplete"
        report["fatal"] = None


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    variants = select_variants(args.variants)
    rank, world, device = setup_distributed()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model_init_seed = args.seed + 30_000
    adapter_init_seed = args.seed + 40_000
    model_id = resolve(args.model)
    requested_revision = args.revision or "main"
    revision_result = None
    if rank == 0:
        try:
            revision_result = {
                "commit": resolve_model_revision(model_id, args.revision),
                "error": None,
            }
        except Exception as exc:
            revision_result = {
                "commit": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    revision_result = broadcast_object(revision_result, rank)
    if revision_result["error"]:
        raise RuntimeError(revision_result["error"])
    resolved_revision = revision_result["commit"]
    report = {
        "schema_version": 2,
        "benchmark": "a100-causal-training-kernels",
        "status": "incomplete",
        "planned_variants": [variant.name for variant in variants],
        "completed_variants": [],
        "fatal": None,
        "supported_evidence_scope": {
            "fused-linear-ce-sdpa": {
                "tuning": "lora",
                "shard": "ddp",
                "peft_version": PEFT_VERSION,
                "forward_source_sha256": LIGER_QWEN2_SOURCE_SHA256,
                "distributed_policy": (
                    "replicated-frozen-base-manual-adapter-gradient-mean"
                ),
                "excluded_until_separate_cuda_evidence": ["full", "fsdp"],
            }
        },
        "trial": {
            "index": args.trial_index,
            "timing_trials_in_record": 1,
            "aggregation": "aggregate separate JSON records by trial.index",
        },
        "environment": environment_report(args, world, device),
        "model": {
            "requested": args.model,
            "model_id": model_id,
            "requested_revision": requested_revision,
            "resolved_commit": resolved_revision,
            "model_init_seed": model_init_seed,
            "adapter_init_seed": adapter_init_seed if args.tuning == "lora" else None,
            "tuning_requested": {
                "mode": args.tuning,
                "lora_r": args.lora_r if args.tuning == "lora" else None,
                "lora_alpha": args.lora_alpha if args.tuning == "lora" else None,
                "lora_targets": args.lora_targets if args.tuning == "lora" else None,
            },
        },
        "variants": [],
    }
    reference_loss = None
    reference_gradients = None
    reference_parameter_deltas = None
    construction_reference_digest = None
    parity_anchor_state = None
    parity_anchor_digest = None
    failed = False
    fatal_phase = None
    fatal_reason = None

    for variant in variants:
        if dist.is_initialized():
            dist.barrier()
        setup_started = time.perf_counter()
        model = None
        tuning_report = None
        kernel_application = None
        error = None
        fatal_load_error = False
        try:
            torch.manual_seed(model_init_seed)
            torch.cuda.manual_seed_all(model_init_seed)
            model, tuning_report, kernel_application = load_raw_model(
                model_id,
                resolved_revision,
                variant,
                dtype,
                device,
                args.tuning,
                args.lora_r,
                args.lora_alpha,
                args.lora_targets,
                adapter_init_seed,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            fatal_load_error = is_fatal_model_load_error(exc)
        loaded_everywhere = all_ranks_succeeded(model is not None, device)
        fatal_load_error = any_rank_true(fatal_load_error, device)
        errors = gather_errors(error, world)
        if not loaded_everywhere:
            if model is not None:
                del model
            model = None
            # Always collect after a load failure, including the rank whose
            # loader raised before returning a model. An instance-bound
            # forward creates a self-cycle that otherwise survives until a
            # later, unrelated arm.
            cleanup_cuda()
            load_reason = errors[0] if errors else "model load failed on another rank"
            record = {
                "variant": asdict(variant),
                "status": (
                    "failed"
                    if variant == REFERENCE_VARIANT or fatal_load_error
                    else "skipped"
                ),
                "reason": load_reason,
            }
            if rank == 0:
                report["variants"].append(record)
            if variant == REFERENCE_VARIANT or fatal_load_error:
                failed = True
                fatal_phase = "model_load"
                fatal_reason = load_reason
                break
            continue

        from torch.nn.parallel import DistributedDataParallel

        # Production LoRA keeps the frozen base unwrapped and manually reduces
        # only replicated trainable adapter gradients. Full tuning retains DDP.
        if world > 1 and args.tuning == "full":
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        construction_digest = trainable_state_digest(model)
        construction_state_report = gather_state_digest_diagnostics(
            construction_digest,
            rank,
            world,
            reference_digest=construction_reference_digest,
        )
        if construction_reference_digest is None:
            construction_reference_digest = construction_state_report[
                "reference_digest"
            ]
        if not construction_state_report["passed"]:
            state_reason = (
                "constructed trainable state did not match the native-SDPA "
                "reference; check model/adapter initialization seeding"
            )
            record = {
                "variant": asdict(variant),
                "status": "failed",
                "reason": state_reason,
                "tuning": tuning_report,
                "kernel_application": kernel_application,
                "construction_trainable_state_sha256": construction_digest,
                "reference_construction_trainable_state_sha256": (
                    construction_reference_digest
                ),
                "state_digest_reports": {
                    "construction": construction_state_report,
                },
            }
            if rank == 0:
                report["variants"].append(record)
            del model
            cleanup_cuda()
            failed = True
            fatal_phase = "construction_state_validation"
            fatal_reason = state_reason
            break

        vocab_size = int(unwrap(model).config.vocab_size)
        controlled_warmup_report = None
        if parity_anchor_state is None:
            if variant != REFERENCE_VARIANT:
                raise RuntimeError("the first benchmark variant must establish the anchor")
            if args.tuning == "lora":
                warm_ids, warm_weights = make_batch(
                    vocab_size,
                    args.parity_micro_batch_size,
                    args.parity_seq_len,
                    args.target_fraction,
                    args.seed + 50_000 + rank,
                    device,
                )
                warm_optimizer = make_optimizer(
                    model, args.learning_rate, args.weight_decay
                )
                torch.manual_seed(args.seed + 60_000 + rank)
                torch.cuda.manual_seed(args.seed + 60_000 + rank)
                if dist.is_initialized():
                    dist.barrier()
                torch.cuda.synchronize(device)
                warm_started = time.perf_counter()
                warm_backward = training_step(
                    model,
                    warm_optimizer,
                    REFERENCE_VARIANT,
                    warm_ids,
                    warm_weights,
                    args.tuning,
                    world,
                    device,
                )
                torch.cuda.synchronize(device)
                warm_seconds = distributed_max(
                    time.perf_counter() - warm_started, device
                )
                del warm_optimizer
                model.zero_grad(set_to_none=True)
                parity_anchor_state = snapshot_trainable_state(model)
                factor_report = lora_factor_nonzero_report(parity_anchor_state)
                controlled_warmup_report = {
                    "performed": True,
                    "source_variant": REFERENCE_VARIANT.name,
                    "batch_seed_base": args.seed + 50_000,
                    "forward_seed_base": args.seed + 60_000,
                    "per_rank_seed_rule": "seed_base + rank",
                    "seconds": warm_seconds,
                    "backward": warm_backward,
                    "lora_factors": factor_report,
                }
            else:
                parity_anchor_state = snapshot_trainable_state(model)
                controlled_warmup_report = {
                    "performed": False,
                    "reason": "controlled adapter warmup applies only to LoRA",
                }
            restored_anchor_digest = trainable_state_digest(model)
            anchor_state_report = gather_state_digest_diagnostics(
                restored_anchor_digest, rank, world
            )
            parity_anchor_digest = anchor_state_report["reference_digest"]
            report["model"]["parity_anchor"] = {
                "trainable_state_sha256": parity_anchor_digest,
                "controlled_warmup": controlled_warmup_report,
                "state_digest_report": anchor_state_report,
            }
        else:
            restore_trainable_state(model, parity_anchor_state)
            restored_anchor_digest = trainable_state_digest(model)
            anchor_state_report = gather_state_digest_diagnostics(
                restored_anchor_digest,
                rank,
                world,
                reference_digest=parity_anchor_digest,
            )

        if not anchor_state_report["passed"]:
            anchor_reason = (
                "restored trainable parity anchor did not match the native-SDPA "
                "warmed state"
            )
            record = {
                "variant": asdict(variant),
                "status": "failed",
                "reason": anchor_reason,
                "tuning": tuning_report,
                "kernel_application": kernel_application,
                "restored_trainable_state_sha256": restored_anchor_digest,
                "parity_anchor_trainable_state_sha256": parity_anchor_digest,
                "state_digest_reports": {
                    "construction": construction_state_report,
                    "warmed_anchor_validation": anchor_state_report,
                },
            }
            if rank == 0:
                report["variants"].append(record)
            del model
            cleanup_cuda()
            failed = True
            fatal_phase = "parity_anchor_restore"
            fatal_reason = anchor_reason
            break

        setup_seconds = distributed_max(time.perf_counter() - setup_started, device)
        parity_ids, parity_weights = make_batch(
            vocab_size,
            args.parity_micro_batch_size,
            args.parity_seq_len,
            args.target_fraction,
            args.seed + rank,
            device,
        )
        parity_started = time.perf_counter()

        # Witness one: exact warmed anchor, fresh optimizer, then exact restore.
        optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
        torch.manual_seed(args.seed + 20_000 + rank)
        torch.cuda.manual_seed(args.seed + 20_000 + rank)
        loss_value, signature, compile_seconds, backward_report = parity_witness(
            model,
            variant,
            parity_ids,
            parity_weights,
            args.tuning,
            world,
            device,
        )
        compile_seconds = distributed_max(compile_seconds, device)
        parameter_deltas, optimizer_init_seconds = parameter_delta_witness(
            model,
            optimizer,
            device,
            restore_parameters=True,
        )
        optimizer_init_seconds = distributed_max(optimizer_init_seconds, device)
        del optimizer
        first_witness_restored_digest = trainable_state_digest(model)
        first_witness_state_report = gather_state_digest_diagnostics(
            first_witness_restored_digest,
            rank,
            world,
            reference_digest=parity_anchor_digest,
        )
        first_witness_restore_failing_ranks = first_witness_state_report[
            "failing_ranks"
        ]
        # Restore explicitly even though the witness performs its own restore;
        # this makes the second arm's starting state an independently verified
        # contract rather than an assumption about the helper above.
        restore_trainable_state(model, parity_anchor_state)

        # Witness two: every arm receives the same independent repeat history.
        optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
        repeat_started = time.perf_counter()
        torch.manual_seed(args.seed + 20_000 + rank)
        torch.cuda.manual_seed(args.seed + 20_000 + rank)
        (
            repeat_loss,
            repeat_signature,
            repeat_forward_backward_seconds,
            repeat_backward_report,
        ) = parity_witness(
            model,
            variant,
            parity_ids,
            parity_weights,
            args.tuning,
            world,
            device,
        )
        repeat_parameter_deltas, repeat_optimizer_step_seconds = (
            parameter_delta_witness(
                model, optimizer, device, restore_parameters=True
            )
        )
        del optimizer
        repeat_seconds = distributed_max(
            time.perf_counter() - repeat_started, device
        )
        repeat_forward_backward_seconds = distributed_max(
            repeat_forward_backward_seconds, device
        )
        repeat_optimizer_step_seconds = distributed_max(
            repeat_optimizer_step_seconds, device
        )

        self_repeat_parity = compare_parity(
            repeat_loss,
            repeat_signature,
            loss_value,
            signature,
            args.parity_rtol,
            args.parity_atol,
            parameter_deltas=repeat_parameter_deltas,
            reference_parameter_deltas=parameter_deltas,
            parameter_delta_rtol=args.parameter_delta_rtol,
            parameter_delta_atol=args.parameter_delta_atol,
        )
        self_repeat_diagnostics = gather_parity_diagnostics(
            self_repeat_parity, rank, world
        )
        self_repeat_parity = apply_distributed_parity(
            self_repeat_parity, self_repeat_diagnostics, rank
        )

        is_reference = reference_loss is None
        if is_reference:
            reference_loss = loss_value
            reference_gradients = signature
            reference_parameter_deltas = parameter_deltas
            parity = self_repeat_parity
            parity["reference_anchor_comparison"] = True
        else:
            parity = compare_parity(
                loss_value,
                signature,
                reference_loss,
                reference_gradients,
                args.parity_rtol,
                args.parity_atol,
                parameter_deltas=parameter_deltas,
                reference_parameter_deltas=reference_parameter_deltas,
                parameter_delta_rtol=args.parameter_delta_rtol,
                parameter_delta_atol=args.parameter_delta_atol,
            )
            anchor_diagnostics = gather_parity_diagnostics(parity, rank, world)
            parity = apply_distributed_parity(parity, anchor_diagnostics, rank)
            parity["reference_anchor_comparison"] = True
            parity["self_repeat_parity"] = self_repeat_parity
            if not self_repeat_parity["passed"]:
                parity["passed"] = False
                parity["passed_all_ranks"] = False
                parity["reason"] = (
                    "self-repeat control failed: "
                    f"{self_repeat_parity['reason']}"
                )

        anchor_failing_ranks = parity["distributed_parity"]["failing_ranks"]
        self_repeat_failing_ranks = self_repeat_parity["distributed_parity"][
            "failing_ranks"
        ]
        parity["overall_failing_ranks"] = sorted(
            set(anchor_failing_ranks) | set(self_repeat_failing_ranks)
        )
        parity["overall_failure_controls"] = {
            "reference_anchor": {
                "passed": parity["distributed_parity"]["passed"],
                "failing_ranks": anchor_failing_ranks,
                "reason": parity["distributed_parity"]["worst_failure_reason"],
            },
            "self_repeat": {
                "passed": self_repeat_parity["distributed_parity"]["passed"],
                "failing_ranks": self_repeat_failing_ranks,
                "reason": self_repeat_parity["distributed_parity"][
                    "worst_failure_reason"
                ],
            },
        }
        parity["first_witness_restore_failing_ranks"] = (
            first_witness_restore_failing_ranks
        )
        parity["first_witness_restored_trainable_state_sha256"] = (
            first_witness_restored_digest
        )
        if first_witness_restore_failing_ranks:
            parity["passed"] = False
            parity["passed_all_ranks"] = False
            parity["reason"] = (
                "first witness did not restore the warmed trainable anchor exactly"
            )
            parity["overall_failing_ranks"] = sorted(
                set(parity["overall_failing_ranks"])
                | set(first_witness_restore_failing_ranks)
            )

        restore_trainable_state(model, parity_anchor_state)
        timing_anchor_digest = trainable_state_digest(model)
        timing_anchor_state_report = gather_state_digest_diagnostics(
            timing_anchor_digest,
            rank,
            world,
            reference_digest=parity_anchor_digest,
        )
        timing_anchor_failing_ranks = timing_anchor_state_report["failing_ranks"]
        timing_anchor_matches_all_ranks = timing_anchor_state_report["passed"]
        if not timing_anchor_matches_all_ranks:
            parity["passed"] = False
            parity["passed_all_ranks"] = False
            parity["reason"] = (
                "failed to restore the identical warmed trainable anchor before timing"
            )
            parity["overall_failing_ranks"] = sorted(
                set(parity["overall_failing_ranks"])
                | set(timing_anchor_failing_ranks)
            )

        parity_seconds = distributed_max(time.perf_counter() - parity_started, device)
        parity_passed = parity["passed"]
        parity["seconds"] = parity_seconds
        parity["first_forward_backward_compile_seconds"] = compile_seconds
        parity["optimizer_init_step_seconds"] = optimizer_init_seconds
        parity["first_backward"] = backward_report
        parity["self_repeat_seconds"] = repeat_seconds
        parity["self_repeat_forward_backward_seconds"] = (
            repeat_forward_backward_seconds
        )
        parity["self_repeat_optimizer_step_seconds"] = repeat_optimizer_step_seconds
        parity["self_repeat_backward"] = repeat_backward_report
        parity["timing_anchor_restored_all_ranks"] = timing_anchor_matches_all_ranks
        parity["timing_anchor_restore_failing_ranks"] = (
            timing_anchor_failing_ranks
        )
        parity["timing_anchor_trainable_state_sha256"] = timing_anchor_digest
        parity["state_digest_reports"] = {
            "construction": construction_state_report,
            "warmed_anchor_validation": anchor_state_report,
            "first_witness_restore": first_witness_state_report,
            "timing_anchor_restore": timing_anchor_state_report,
        }
        if not parity_passed:
            if parity["reason"] is None:
                parity["reason"] = "parity gate failed on another rank"
            record = {
                "variant": asdict(variant),
                "status": "failed",
                "setup_seconds": setup_seconds,
                "tuning": tuning_report,
                "kernel_application": kernel_application,
                "model_init_seed": model_init_seed,
                "adapter_init_seed": (
                    adapter_init_seed if args.tuning == "lora" else None
                ),
                "construction_trainable_state_sha256": construction_digest,
                "reference_construction_trainable_state_sha256": (
                    construction_reference_digest
                ),
                "parity_anchor_trainable_state_sha256": parity_anchor_digest,
                "state_digest_reports": parity["state_digest_reports"],
                "parity": parity,
            }
            if rank == 0:
                report["variants"].append(record)
            del model
            cleanup_cuda()
            failed = True
            fatal_phase = "parity"
            fatal_reason = parity["reason"]
            break

        if not is_reference:
            # The full reference snapshots stay resident for later variants;
            # a passing candidate snapshot is no longer needed during timing.
            del signature
            del parameter_deltas
        del repeat_signature
        del repeat_parameter_deltas

        optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
        input_ids, weights = make_batch(
            vocab_size,
            args.micro_batch_size,
            args.seq_len,
            args.target_fraction,
            args.seed + 10_000 + rank,
            device,
        )
        metrics = benchmark_variant(
            model,
            optimizer,
            variant,
            input_ids,
            weights,
            args.warmup_steps,
            args.steps,
            args.tuning,
            world,
            device,
        )
        if rank == 0:
            report["variants"].append(
                {
                    "variant": asdict(variant),
                    "status": "passed",
                    "setup_seconds": setup_seconds,
                    "tuning": tuning_report,
                    "kernel_application": kernel_application,
                    "model_init_seed": model_init_seed,
                    "adapter_init_seed": (
                        adapter_init_seed if args.tuning == "lora" else None
                    ),
                    "construction_trainable_state_sha256": construction_digest,
                    "reference_construction_trainable_state_sha256": (
                        construction_reference_digest
                    ),
                    "parity_anchor_trainable_state_sha256": parity_anchor_digest,
                    "state_digest_reports": parity["state_digest_reports"],
                    "parity": parity,
                    "metrics": metrics,
                }
            )
        del optimizer
        del model
        cleanup_cuda()

    if rank == 0:
        finalize_report_status(report, failed, fatal_phase, fatal_reason)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        args.output.write_text(serialized + "\n")
        print(serialized, flush=True)
        print(f"wrote {args.output}", file=sys.stderr, flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
