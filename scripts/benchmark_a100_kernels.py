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
import json
import math
import os
import platform
import socket
import statistics
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
    attention_load_kwargs,
    liger_sft_forward,
    require_liger_model_support,
    resolved_attention_backend,
    validate_kernel_request,
)
from yeto.losses import sft_loss  # noqa: E402
from yeto.models import resolve  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    attention_backend: str
    kernel_backend: str
    loss_implementation: str


VARIANTS = (
    Variant("native-sdpa", "sdpa", "native", "torch-fused-cross-entropy"),
    Variant(
        "native-flash-attn-2",
        "flash-attn-2",
        "native",
        "torch-fused-cross-entropy",
    ),
    Variant("liger-sdpa", "sdpa", "liger", "liger-fused-linear-cross-entropy"),
    Variant(
        "liger-flash-attn-2",
        "flash-attn-2",
        "liger",
        "liger-fused-linear-cross-entropy",
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
):
    from transformers import AutoConfig, AutoModelForCausalLM

    validate_kernel_request(
        variant.kernel_backend,
        "cross_entropy",
        device,
        dtype,
    )
    kwargs = attention_load_kwargs(variant.attention_backend, device, dtype)
    factory = AutoModelForCausalLM
    if variant.kernel_backend == "liger":
        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )
        require_liger_model_support(config)
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        factory = AutoLigerKernelForCausalLM
        kwargs.update(cross_entropy=False, fused_linear_cross_entropy=True)
    model = factory.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=True,
        **kwargs,
    )
    model.to(device)
    model.train()
    model.config.use_cache = False
    resolved_attention_backend(model, variant.attention_backend)
    return model


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
    if variant.kernel_backend == "liger":
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
    for name, parameter in unwrap(model).named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        snapshot[name] = parameter.grad.detach().cpu().clone()
    return snapshot


def tensor_parity(actual, reference, rtol, atol, chunk_size=1_000_000):
    if actual.shape != reference.shape:
        return False, math.inf, math.inf
    actual = actual.reshape(-1)
    reference = reference.reshape(-1)
    passed = True
    max_absolute = 0.0
    max_relative = 0.0
    for start in range(0, actual.numel(), chunk_size):
        stop = min(start + chunk_size, actual.numel())
        actual_chunk = actual[start:stop].float()
        reference_chunk = reference[start:stop].float()
        difference = (actual_chunk - reference_chunk).abs()
        if difference.numel():
            max_absolute = max(max_absolute, float(difference.max().item()))
            relative = difference / reference_chunk.abs().clamp_min(1e-12)
            max_relative = max(max_relative, float(relative.max().item()))
        if not torch.allclose(actual_chunk, reference_chunk, rtol=rtol, atol=atol):
            passed = False
    return passed, max_absolute, max_relative


def compare_parity(
    loss: float,
    gradients: dict[str, torch.Tensor],
    reference_loss: float,
    reference_gradients: dict[str, torch.Tensor],
    rtol: float,
    atol: float,
    parameter_deltas: dict[str, torch.Tensor] | None = None,
    reference_parameter_deltas: dict[str, torch.Tensor] | None = None,
) -> dict:
    result = {
        "passed": True,
        "loss": loss,
        "reference_loss": reference_loss,
        "loss_abs_error": abs(loss - reference_loss),
        "max_gradient_abs_error": 0.0,
        "max_gradient_relative_error": 0.0,
        "checked_gradient_tensors": len(reference_gradients),
        "max_parameter_delta_abs_error": 0.0,
        "max_parameter_delta_relative_error": 0.0,
        "checked_parameter_delta_tensors": (
            len(reference_parameter_deltas) if reference_parameter_deltas else 0
        ),
        "reason": None,
    }
    if gradients.keys() != reference_gradients.keys():
        missing = sorted(reference_gradients.keys() - gradients.keys())
        extra = sorted(gradients.keys() - reference_gradients.keys())
        result.update(
            passed=False,
            reason=f"gradient key mismatch: missing={missing[:5]} extra={extra[:5]}",
        )
        return result
    if not math.isclose(loss, reference_loss, rel_tol=rtol, abs_tol=atol):
        result.update(passed=False, reason="loss parity gate failed")
    for name, reference in reference_gradients.items():
        actual = gradients[name]
        tensor_passed, absolute, relative = tensor_parity(
            actual, reference, rtol, atol
        )
        result["max_gradient_abs_error"] = max(
            result["max_gradient_abs_error"], absolute
        )
        result["max_gradient_relative_error"] = max(
            result["max_gradient_relative_error"], relative
        )
        if not tensor_passed:
            result.update(
                passed=False,
                reason=f"gradient parity gate failed for {name}",
            )
            break
    if not result["passed"]:
        return result
    if (parameter_deltas is None) != (reference_parameter_deltas is None):
        result.update(passed=False, reason="parameter-delta parity inputs are incomplete")
        return result
    if parameter_deltas is not None:
        if parameter_deltas.keys() != reference_parameter_deltas.keys():
            missing = sorted(reference_parameter_deltas.keys() - parameter_deltas.keys())
            extra = sorted(parameter_deltas.keys() - reference_parameter_deltas.keys())
            result.update(
                passed=False,
                reason=(
                    f"parameter-delta key mismatch: missing={missing[:5]} "
                    f"extra={extra[:5]}"
                ),
            )
            return result
        for name, reference in reference_parameter_deltas.items():
            delta_passed, absolute, relative = tensor_parity(
                parameter_deltas[name], reference, rtol, atol
            )
            result["max_parameter_delta_abs_error"] = max(
                result["max_parameter_delta_abs_error"], absolute
            )
            result["max_parameter_delta_relative_error"] = max(
                result["max_parameter_delta_relative_error"], relative
            )
            if not delta_passed:
                result.update(
                    passed=False,
                    reason=f"parameter-delta parity gate failed for {name}",
                )
                break
    return result


def parity_witness(model, variant, input_ids, weights, device):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss, target_tokens = forward_sum(model, variant, input_ids, weights)
    (loss / target_tokens.clamp(min=1)).backward()
    torch.cuda.synchronize(device)
    compile_seconds = time.perf_counter() - started
    witness = gradient_snapshot(model)
    value = float(loss.detach().float().item())
    return value, witness, compile_seconds


def parameter_delta_witness(model, optimizer, device):
    """Apply one optimizer step and retain every trainable parameter delta."""
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
        deltas[name] = after.float() - before.pop(name).float()
    if before:
        raise RuntimeError(f"trainable parameters disappeared during optimizer step: {list(before)[:5]}")
    optimizer.zero_grad(set_to_none=True)
    return deltas, optimizer_step_seconds


def training_step(model, optimizer, variant, input_ids, weights) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss, target_tokens = forward_sum(model, variant, input_ids, weights)
    (loss / target_tokens.clamp(min=1)).backward()
    optimizer.step()


def benchmark_variant(
    model,
    optimizer,
    variant,
    input_ids,
    weights,
    warmup_steps: int,
    measured_steps: int,
    world: int,
    device: torch.device,
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    training_step(model, optimizer, variant, input_ids, weights)
    torch.cuda.synchronize(device)
    first_optimizer_step_seconds = distributed_max(time.perf_counter() - started, device)

    for _ in range(warmup_steps):
        training_step(model, optimizer, variant, input_ids, weights)
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()

    durations: list[float] = []
    for _ in range(measured_steps):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        training_step(model, optimizer, variant, input_ids, weights)
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
    parser.add_argument("--variants", default="all", help="all or comma-separated variant names")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--parity-micro-batch-size", type=int, default=1)
    parser.add_argument("--parity-seq-len", type=int, default=128)
    parser.add_argument("--target-fraction", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
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


def environment_report(args, world, device) -> dict:
    capability = torch.cuda.get_device_capability(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
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


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    variants = select_variants(args.variants)
    rank, world, device = setup_distributed()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
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
        "schema_version": 1,
        "benchmark": "a100-causal-training-kernels",
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
        },
        "variants": [],
    }
    reference_loss = None
    reference_gradients = None
    reference_parameter_deltas = None
    failed = False

    for variant in variants:
        if dist.is_initialized():
            dist.barrier()
        setup_started = time.perf_counter()
        model = None
        error = None
        fatal_load_error = False
        try:
            model = load_raw_model(
                model_id,
                resolved_revision,
                variant,
                dtype,
                device,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            fatal_load_error = isinstance(exc, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(exc).lower()
            )
        loaded_everywhere = all_ranks_succeeded(model is not None, device)
        fatal_load_error = any_rank_true(fatal_load_error, device)
        errors = gather_errors(error, world)
        if not loaded_everywhere:
            if model is not None:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            record = {
                "variant": asdict(variant),
                "status": (
                    "failed"
                    if variant == REFERENCE_VARIANT or fatal_load_error
                    else "skipped"
                ),
                "reason": errors[0] if errors else "model load failed on another rank",
            }
            if rank == 0:
                report["variants"].append(record)
            if variant == REFERENCE_VARIANT or fatal_load_error:
                failed = True
                break
            continue

        from torch.nn.parallel import DistributedDataParallel

        if world > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=False,
            fused=False,
        )
        setup_seconds = distributed_max(time.perf_counter() - setup_started, device)
        vocab_size = int(unwrap(model).config.vocab_size)
        parity_ids, parity_weights = make_batch(
            vocab_size,
            args.parity_micro_batch_size,
            args.parity_seq_len,
            args.target_fraction,
            args.seed + rank,
            device,
        )
        parity_started = time.perf_counter()
        torch.manual_seed(args.seed + 20_000 + rank)
        torch.cuda.manual_seed(args.seed + 20_000 + rank)
        loss_value, signature, compile_seconds = parity_witness(
            model, variant, parity_ids, parity_weights, device
        )
        compile_seconds = distributed_max(compile_seconds, device)
        parameter_deltas, optimizer_init_seconds = parameter_delta_witness(
            model, optimizer, device
        )
        optimizer_init_seconds = distributed_max(optimizer_init_seconds, device)
        parity_seconds = distributed_max(time.perf_counter() - parity_started, device)
        if reference_loss is None:
            reference_loss = loss_value
            reference_gradients = signature
            reference_parameter_deltas = parameter_deltas
            parity = {
                "passed": True,
                "loss": loss_value,
                "reference_loss": loss_value,
                "loss_abs_error": 0.0,
                "max_gradient_abs_error": 0.0,
                "max_gradient_relative_error": 0.0,
                "checked_gradient_tensors": len(signature),
                "max_parameter_delta_abs_error": 0.0,
                "max_parameter_delta_relative_error": 0.0,
                "checked_parameter_delta_tensors": len(parameter_deltas),
                "reason": None,
            }
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
            )
        parity_passed = all_ranks_succeeded(parity["passed"], device)
        parity["passed_all_ranks"] = parity_passed
        parity["seconds"] = parity_seconds
        parity["first_forward_backward_compile_seconds"] = compile_seconds
        parity["optimizer_init_step_seconds"] = optimizer_init_seconds
        if not parity_passed:
            if parity["reason"] is None:
                parity["reason"] = "parity gate failed on another rank"
            record = {
                "variant": asdict(variant),
                "status": "failed",
                "setup_seconds": setup_seconds,
                "parity": parity,
            }
            if rank == 0:
                report["variants"].append(record)
            del optimizer
            del model
            cleanup_cuda()
            failed = True
            break

        if variant != REFERENCE_VARIANT:
            # The full reference snapshots stay resident for later variants;
            # a passing candidate snapshot is no longer needed during timing.
            del signature
            del parameter_deltas

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
            world,
            device,
        )
        if rank == 0:
            report["variants"].append(
                {
                    "variant": asdict(variant),
                    "status": "passed",
                    "setup_seconds": setup_seconds,
                    "parity": parity,
                    "metrics": metrics,
                }
            )
        del optimizer
        del model
        cleanup_cuda()

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print(f"wrote {args.output}", file=sys.stderr, flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
