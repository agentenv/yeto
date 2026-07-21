#!/usr/bin/env python3
"""Parity-gated causal-LM kernel benchmark for one 8xA100 node.

This script never provisions infrastructure. Run it on an existing node with
``torchrun --standalone --nproc_per_node=8``. Optional combinations that are
not installed are recorded as skipped; a parity failure is fatal and is never
converted into a performance result.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
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
import tempfile
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Iterable

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
from yeto.learner import allreduce_trainable_grads, resolve_lora_targets  # noqa: E402
from yeto.models import resolve  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    attention_backend: str
    kernel_backend: str
    loss_implementation: str
    internal_sdpa_backend: str | None


VARIANTS = (
    Variant(
        "native-sdpa",
        "sdpa",
        "native",
        "torch-fused-cross-entropy",
        "auto",
    ),
    Variant(
        "native-sdpa-flash",
        "sdpa",
        "native",
        "torch-fused-cross-entropy",
        "flash",
    ),
    Variant(
        "native-sdpa-math",
        "sdpa",
        "native",
        "torch-fused-cross-entropy",
        "math",
    ),
    Variant(
        "native-sdpa-efficient",
        "sdpa",
        "native",
        "torch-fused-cross-entropy",
        "efficient",
    ),
    Variant(
        "native-sdpa-cudnn",
        "sdpa",
        "native",
        "torch-fused-cross-entropy",
        "cudnn",
    ),
    Variant(
        "native-flash-attn-2",
        "flash-attn-2",
        "native",
        "torch-fused-cross-entropy",
        None,
    ),
    Variant(
        "liger-sdpa",
        "sdpa",
        "liger",
        "liger-fused-linear-cross-entropy",
        "auto",
    ),
    Variant(
        "liger-flash-attn-2",
        "flash-attn-2",
        "liger",
        "liger-fused-linear-cross-entropy",
        None,
    ),
)
VARIANTS_BY_NAME = {variant.name: variant for variant in VARIANTS}
REFERENCE_VARIANT = VARIANTS[0]

SDPA_BACKEND_NAMES = ("auto", "flash", "math", "efficient", "cudnn")
SDPA_FLAG_GETTERS = {
    "flash": "flash_sdp_enabled",
    "math": "math_sdp_enabled",
    "efficient": "mem_efficient_sdp_enabled",
    "cudnn": "cudnn_sdp_enabled",
}
SDPA_FLAG_SETTERS = {
    "flash": "enable_flash_sdp",
    "math": "enable_math_sdp",
    "efficient": "enable_mem_efficient_sdp",
    "cudnn": "enable_cudnn_sdp",
}


def sdpa_backend_objects(name: str) -> list:
    """Map a stable report name to PyTorch's public SDPA selector enum."""
    if name not in SDPA_BACKEND_NAMES:
        raise ValueError(
            f"unknown internal SDPA backend {name!r}; choose from {SDPA_BACKEND_NAMES}"
        )
    from torch.nn.attention import SDPBackend

    mapping = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "math": SDPBackend.MATH,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    if name == "auto":
        # This is PyTorch 2.5.1's public all-enabled behavior. Ordering is not
        # asserted as a priority: the profiler below records the actual choice.
        return [mapping[item] for item in ("flash", "efficient", "math", "cudnn")]
    return [mapping[name]]


def snapshot_sdpa_backend_flags() -> dict[str, bool]:
    """Read every public CUDA SDPA enable flag or fail before benchmarking."""
    flags = {}
    for name, getter_name in SDPA_FLAG_GETTERS.items():
        getter = getattr(torch.backends.cuda, getter_name, None)
        if getter is None:
            raise RuntimeError(f"PyTorch exposes no public {getter_name}() selector")
        flags[name] = bool(getter())
    return flags


def restore_sdpa_backend_flags(flags: dict[str, bool]) -> None:
    if set(flags) != set(SDPA_FLAG_SETTERS):
        raise ValueError("an SDPA flag snapshot must contain every known backend")
    for name, setter_name in SDPA_FLAG_SETTERS.items():
        setter = getattr(torch.backends.cuda, setter_name, None)
        if setter is None:
            raise RuntimeError(f"PyTorch exposes no public {setter_name}() selector")
        setter(flags[name])


def _empty_sdpa_control(name: str | None) -> dict:
    return {
        "api": "torch.nn.attention.sdpa_kernel",
        "requested": name,
        "before": None,
        "expected_active": None,
        "active": None,
        "after": None,
        "restored_exactly": False,
    }


@contextlib.contextmanager
def sdpa_backend_context(name: str | None, control: dict | None = None):
    """Select one internal backend for an entire arm and verify exact restore.

    The context uses only the public PyTorch 2.5.1 selector. A mismatch while
    active or after exit is fatal even if a manual repair succeeds, because a
    publishable timing cannot rely on silently leaked process-global flags.
    """
    if control is None:
        control = _empty_sdpa_control(name)
    elif control.get("requested") != name:
        raise ValueError("the SDPA control record does not match the requested backend")

    before = snapshot_sdpa_backend_flags()
    control["before"] = before
    expected = (
        before
        if name is None
        else {
            backend: name == "auto" or backend == name for backend in SDPA_FLAG_GETTERS
        }
    )
    control["expected_active"] = expected
    selector = contextlib.nullcontext()
    if name is not None:
        from torch.nn.attention import sdpa_kernel

        selector = sdpa_kernel(sdpa_backend_objects(name))

    restore_error = None
    try:
        with selector:
            active = snapshot_sdpa_backend_flags()
            control["active"] = active
            if active != expected:
                raise RuntimeError(
                    f"public SDPA selector activated {active}, expected {expected}"
                )
            yield control
    finally:
        after = snapshot_sdpa_backend_flags()
        control["after"] = after
        control["restored_exactly"] = after == before
        if after != before:
            try:
                restore_sdpa_backend_flags(before)
            except Exception as exc:  # preserve repair evidence in the failure
                restore_error = f"{type(exc).__name__}: {exc}"
            repaired = snapshot_sdpa_backend_flags()
            control["repair_attempted"] = True
            control["repair_error"] = restore_error
            control["after_repair"] = repaired
            raise RuntimeError(
                "public sdpa_kernel context did not restore every backend flag "
                f"exactly: before={before} after={after} repaired={repaired}"
            )


class SDPAArmController:
    """Keep the selector active across one complete arm without a giant indent."""

    def __init__(self):
        self._context = None
        self._control = None

    @property
    def active(self) -> bool:
        return self._context is not None

    @property
    def control(self) -> dict | None:
        return self._control

    def activate(self, variant: Variant) -> dict:
        if self._context is not None:
            raise RuntimeError(
                "an SDPA arm is already active; restore it explicitly before activation"
            )
        control = _empty_sdpa_control(variant.internal_sdpa_backend)
        context = sdpa_backend_context(variant.internal_sdpa_backend, control)
        self._control = control
        try:
            context.__enter__()
        except BaseException:
            self._context = None
            raise
        self._context = context
        return control

    def close(self, exc_info=(None, None, None)) -> dict | None:
        if self._context is None:
            return self._control
        context, self._context = self._context, None
        context.__exit__(*exc_info)
        return self._control


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


def select_variants(
    spec: str,
    reference_name: str = REFERENCE_VARIANT.name,
) -> list[Variant]:
    if reference_name not in VARIANTS_BY_NAME:
        raise ValueError(
            f"unknown reference variant {reference_name!r}; choose from "
            f"{list(VARIANTS_BY_NAME)}"
        )
    reference = VARIANTS_BY_NAME[reference_name]
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if "all" in names:
        raise ValueError(
            "publishable runs accept the selected reference plus at most one "
            "candidate; select one candidate instead of 'all'"
        )
    unknown = [name for name in names if name not in VARIANTS_BY_NAME]
    if unknown:
        raise ValueError(
            f"unknown variants {unknown}; choose from {list(VARIANTS_BY_NAME)}"
        )
    candidates = set(names) - {reference.name}
    if len(candidates) > 1:
        raise ValueError(
            "publishable runs accept the selected reference plus at most one "
            f"candidate, received {sorted(candidates)}"
        )
    return [reference] + [variant for variant in VARIANTS if variant.name in candidates]


SDPA_PRIMARY_OPERATORS = {
    "flash": "aten::_scaled_dot_product_flash_attention",
    "math": "aten::_scaled_dot_product_attention_math",
    "efficient": "aten::_scaled_dot_product_efficient_attention",
    "cudnn": "aten::_scaled_dot_product_cudnn_attention",
}
SDPA_GENERIC_OPERATOR = "aten::scaled_dot_product_attention"
SDPA_FORWARD_ATEN_ALLOWLIST = frozenset(
    {SDPA_GENERIC_OPERATOR, *SDPA_PRIMARY_OPERATORS.values()}
)


def _strict_json_sha256(value) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tensor_signature(tensor: torch.Tensor | None) -> dict | None:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"expected an SDPA tensor input, received {type(tensor).__name__}"
        )
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device_type": tensor.device.type,
        "device_index": tensor.device.index,
        "layout": str(tensor.layout).removeprefix("torch."),
        "requires_grad": bool(tensor.requires_grad),
        "is_contiguous": bool(tensor.is_contiguous()),
        "is_nested": bool(getattr(tensor, "is_nested", False)),
        "storage_offset": int(tensor.storage_offset()),
    }


def sdpa_selector_eligibility(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    enable_gqa: bool,
) -> dict[str, bool]:
    """Evaluate each public fused-backend predicate on the exact call inputs."""
    params_type = getattr(torch.backends.cuda, "SDPAParams", None)
    if params_type is None:
        raise RuntimeError("PyTorch exposes no public torch.backends.cuda.SDPAParams")
    params = params_type(
        query,
        key,
        value,
        attn_mask,
        float(dropout_p),
        bool(is_causal),
        bool(enable_gqa),
    )
    checks = {
        "flash": "can_use_flash_attention",
        "efficient": "can_use_efficient_attention",
        "cudnn": "can_use_cudnn_attention",
    }
    result = {"math": True}
    for backend, check_name in checks.items():
        check = getattr(torch.backends.cuda, check_name, None)
        if check is None:
            raise RuntimeError(f"PyTorch exposes no public {check_name}() predicate")
        result[backend] = bool(check(params, debug=False))
    return {name: result[name] for name in SDPA_FLAG_GETTERS}


def _sdpa_call_arguments(args: tuple, kwargs: dict) -> dict:
    names = (
        "query",
        "key",
        "value",
        "attn_mask",
        "dropout_p",
        "is_causal",
        "scale",
        "enable_gqa",
    )
    if len(args) > len(names):
        raise TypeError("scaled_dot_product_attention received too many arguments")
    values = dict(zip(names, args))
    unknown = sorted(set(kwargs) - set(names))
    duplicate = sorted(set(kwargs) & set(values))
    if unknown:
        raise TypeError(f"unknown scaled_dot_product_attention arguments: {unknown}")
    if duplicate:
        raise TypeError(
            f"duplicate scaled_dot_product_attention arguments: {duplicate}"
        )
    values.update(kwargs)
    missing = [name for name in names[:3] if name not in values]
    if missing:
        raise TypeError(f"missing scaled_dot_product_attention arguments: {missing}")
    values.setdefault("attn_mask", None)
    values.setdefault("dropout_p", 0.0)
    values.setdefault("is_causal", False)
    values.setdefault("scale", None)
    values.setdefault("enable_gqa", False)
    return values


def sdpa_input_signature(args: tuple, kwargs: dict) -> dict:
    values = _sdpa_call_arguments(args, kwargs)
    scale = values["scale"]
    if scale is not None:
        scale = float(scale)
    eligibility = sdpa_selector_eligibility(
        values["query"],
        values["key"],
        values["value"],
        values["attn_mask"],
        float(values["dropout_p"]),
        bool(values["is_causal"]),
        bool(values["enable_gqa"]),
    )
    return {
        "query": _tensor_signature(values["query"]),
        "key": _tensor_signature(values["key"]),
        "value": _tensor_signature(values["value"]),
        "attn_mask": _tensor_signature(values["attn_mask"]),
        "dropout_p": float(values["dropout_p"]),
        "is_causal": bool(values["is_causal"]),
        "scale": scale,
        "enable_gqa": bool(values["enable_gqa"]),
        "grad_enabled": bool(torch.is_grad_enabled()),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
        "selector_eligibility": eligibility,
    }


class SDPAInputRecorder:
    """Record every public functional SDPA call without retaining tensors."""

    def __init__(self):
        self._original = None
        self._wrapper = None
        self._signatures: list[dict] = []

    def __enter__(self):
        functional = torch.nn.functional
        if self._original is not None:
            raise RuntimeError("SDPA input recorder cannot be nested or reused")
        original = functional.scaled_dot_product_attention
        self._original = original

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            self._signatures.append(sdpa_input_signature(args, kwargs))
            return original(*args, **kwargs)

        self._wrapper = wrapper
        functional.scaled_dot_product_attention = wrapper
        return self

    def __exit__(self, exc_type, exc, traceback):
        functional = torch.nn.functional
        replaced = functional.scaled_dot_product_attention is not self._wrapper
        original = self._original
        functional.scaled_dot_product_attention = original
        restored = functional.scaled_dot_product_attention is original
        self._original = None
        self._wrapper = None
        if replaced or not restored:
            raise RuntimeError(
                "SDPA input recorder observed an unexpected functional patch or "
                "could not restore the exact original callable"
            )
        return False

    def report(self) -> dict:
        unique: dict[str, dict] = {}
        ordered_digests = []
        for index, signature in enumerate(self._signatures):
            digest = _strict_json_sha256(signature)
            ordered_digests.append(digest)
            if digest not in unique:
                unique[digest] = {
                    "sha256": digest,
                    "first_call_index": index,
                    "call_count": 0,
                    "signature": signature,
                }
            unique[digest]["call_count"] += 1
        return {
            "total_calls": len(self._signatures),
            "unique_signature_count": len(unique),
            "ordered_signature_sha256": _strict_json_sha256(ordered_digests),
            "unique_signatures": sorted(
                unique.values(), key=lambda item: item["first_call_index"]
            ),
        }


def parse_sdpa_profiler_events(events: Iterable) -> dict:
    """Extract public SDPA dispatch evidence from profiler key averages."""
    operator_counts: dict[str, int] = {}
    for event in events:
        if isinstance(event, str):
            key, count = event, 1
        elif isinstance(event, dict):
            key, count = event["key"], event.get("count", 1)
        else:
            key, count = event.key, getattr(event, "count", 1)
        if not isinstance(key, str) or not isinstance(count, int) or count < 0:
            raise ValueError(
                "profiler events require a string key and nonnegative count"
            )
        if "attention" in key or "scaled_dot_product" in key:
            operator_counts[key] = operator_counts.get(key, 0) + count

    aten_attention_operator_counts = {
        key: count
        for key, count in operator_counts.items()
        if key.startswith("aten::") and count > 0
    }
    unexpected_aten_attention_operator_counts = {
        key: count
        for key, count in aten_attention_operator_counts.items()
        if key not in SDPA_FORWARD_ATEN_ALLOWLIST
    }
    backend_counts = {
        backend: operator_counts.get(operator, 0)
        for backend, operator in SDPA_PRIMARY_OPERATORS.items()
    }
    return {
        "generic_sdpa_call_count": operator_counts.get(SDPA_GENERIC_OPERATOR, 0),
        "primary_backend_call_count": sum(backend_counts.values()),
        "backend_call_counts": backend_counts,
        "observed_backends": [
            backend for backend in SDPA_FLAG_GETTERS if backend_counts[backend] > 0
        ],
        "forward_only_aten_allowlist": sorted(SDPA_FORWARD_ATEN_ALLOWLIST),
        "aten_attention_operator_counts": dict(
            sorted(aten_attention_operator_counts.items())
        ),
        "unexpected_aten_attention_operator_counts": dict(
            sorted(unexpected_aten_attention_operator_counts.items())
        ),
        "operator_counts": dict(sorted(operator_counts.items())),
    }


def evaluate_local_sdpa_attribution(
    selector_backend: str | None,
    recorder: dict,
    profiler: dict,
    error: str | None = None,
) -> dict:
    """Apply the no-timing attribution gate for one rank and one input shape."""
    errors = []
    if error:
        errors.append(error)
    recorded_calls = recorder["total_calls"]
    generic_calls = profiler["generic_sdpa_call_count"]
    primary_calls = profiler["primary_backend_call_count"]
    observed = set(profiler["observed_backends"])
    unexpected_aten = profiler.get("unexpected_aten_attention_operator_counts", {})
    if unexpected_aten:
        errors.append(
            "the forward-only profiler observed unexpected ATen attention "
            f"operators: {unexpected_aten}"
        )

    if selector_backend is None:
        if recorded_calls or generic_calls or primary_calls:
            errors.append(
                "the non-SDPA arm unexpectedly executed an internal PyTorch SDPA operator"
            )
        selector_operator_agreement = (
            not recorded_calls and not generic_calls and not primary_calls
        )
        selector_eligibility_all_calls = True
    else:
        allowed = (
            set(SDPA_FLAG_GETTERS) if selector_backend == "auto" else {selector_backend}
        )
        if recorded_calls < 1:
            errors.append("the functional recorder observed no SDPA calls")
        if generic_calls != recorded_calls:
            errors.append(
                "functional/profiler SDPA call coverage differed: "
                f"recorder={recorded_calls} profiler={generic_calls}"
            )
        if primary_calls != recorded_calls:
            errors.append(
                "profiler primary-backend coverage differed from the full input "
                f"record: primary={primary_calls} recorder={recorded_calls}"
            )
        if not observed:
            errors.append("the profiler observed no recognized primary SDPA backend")
        selector_operator_agreement = bool(observed) and observed <= allowed
        if selector_backend != "auto":
            selector_operator_agreement = observed == {selector_backend}
        if not selector_operator_agreement:
            errors.append(
                f"selector {selector_backend!r} disagreed with profiler backends "
                f"{sorted(observed)}"
            )

        eligibility_by_signature = [
            item["signature"]["selector_eligibility"]
            for item in recorder["unique_signatures"]
        ]
        if selector_backend == "auto":
            if len(observed) != 1:
                errors.append(
                    "automatic SDPA attribution requires exactly one observed "
                    f"backend per probe, found {sorted(observed)}"
                )
                selector_eligibility_all_calls = False
            else:
                observed_backend = next(iter(observed))
                selector_eligibility_all_calls = all(
                    eligibility[observed_backend]
                    for eligibility in eligibility_by_signature
                )
        else:
            selector_eligibility_all_calls = all(
                eligibility[selector_backend]
                for eligibility in eligibility_by_signature
            )
        if not selector_eligibility_all_calls:
            errors.append(
                "the public capability predicates did not support the observed "
                "selector for every unique input signature"
            )

    return {
        "selector_backend": selector_backend,
        "selector_is_exact": selector_backend not in (None, "auto"),
        "selector_operator_agreement": selector_operator_agreement,
        "selector_eligibility_all_calls": selector_eligibility_all_calls,
        "passed": not errors,
        "errors": errors,
        "recorder": recorder,
        "profiler": profiler,
    }


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


def exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def gather_rank_records(local: dict, rank: int, world: int) -> list[dict]:
    """Gather one JSON-safe control record and require every expected rank."""
    report = dict(local)
    report["rank"] = rank
    if dist.is_initialized():
        gathered: list[dict | None] = [None] * world
        dist.all_gather_object(gathered, report)
        if any(item is None for item in gathered):
            raise RuntimeError("distributed control-record gather was incomplete")
        reports = [item for item in gathered if item is not None]
    else:
        reports = [report]
    ordered = sorted(reports, key=lambda item: item["rank"])
    ranks = [item["rank"] for item in ordered]
    expected = list(range(world))
    if ranks != expected:
        raise RuntimeError(
            f"distributed control records expected ranks {expected}, got {ranks}"
        )
    return ordered


def aggregate_sdpa_control_phase(
    rank_reports: list[dict],
    expected_world: int,
    variant_name: str,
    phase: str,
) -> dict:
    if phase not in ("activation", "restoration", "final_audit"):
        raise ValueError(f"unknown SDPA control phase {phase!r}")
    ordered = sorted(rank_reports, key=lambda item: item["rank"])
    ranks = [item["rank"] for item in ordered]
    if ranks != list(range(expected_world)) or len(set(ranks)) != len(ranks):
        raise ValueError(
            f"SDPA {phase} expected ranks {list(range(expected_world))}, got {ranks}"
        )
    if any(item["phase"] != phase for item in ordered):
        raise ValueError(f"SDPA {phase} records mixed lifecycle phases")
    if any(item["variant"] != variant_name for item in ordered):
        raise ValueError(f"SDPA {phase} records mixed variants")

    failing_ranks = []
    for item in ordered:
        control = item.get("control")
        passed = item.get("error") is None
        if phase == "activation":
            passed = passed and item["active_after_phase"]
            passed = passed and control is not None
            if control is not None:
                passed = passed and control.get("active") == control.get(
                    "expected_active"
                )
        elif phase == "restoration":
            passed = passed and not item["active_after_phase"]
            restore_was_required = item["had_active_context"] or (
                control is not None and control.get("active") is not None
            )
            item["restore_was_required"] = restore_was_required
            if restore_was_required:
                passed = passed and control is not None
                if control is not None:
                    passed = passed and bool(control.get("restored_exactly"))
        else:
            passed = passed and not item["active_after_phase"]
        item["passed"] = bool(passed)
        if not passed:
            failing_ranks.append(item["rank"])

    return {
        "phase": phase,
        "variant": variant_name,
        "passed": not failing_ranks,
        "failing_ranks": failing_ranks,
        "rank_reports": ordered,
    }


def finish_sdpa_arm(
    controller: SDPAArmController,
    lifecycle: dict,
    rank: int,
    world: int,
) -> dict:
    """Restore one arm locally, then make restoration an all-rank gate."""
    if lifecycle.get("restoration") is not None:
        raise RuntimeError("the SDPA arm has already been restored")
    variant_name = lifecycle["variant"]
    had_active_context = controller.active
    errors = []
    flags_before_close = None
    if had_active_context:
        try:
            flags_before_close = snapshot_sdpa_backend_flags()
            expected = (controller.control or {}).get("expected_active")
            if flags_before_close != expected:
                errors.append(
                    "RuntimeError: SDPA flags changed while the arm was active: "
                    f"expected={expected} before_close={flags_before_close}"
                )
        except Exception as exc:
            errors.append(
                f"SDPA pre-restoration flag snapshot failed: {exception_text(exc)}"
            )
    try:
        controller.close()
    except Exception as exc:
        errors.append(exception_text(exc))
    local = {
        "phase": "restoration",
        "variant": variant_name,
        "had_active_context": had_active_context,
        "flags_before_close": flags_before_close,
        "active_after_phase": controller.active,
        "error": "; ".join(errors) if errors else None,
        "control": json.loads(json.dumps(controller.control, allow_nan=False))
        if controller.control is not None
        else None,
    }
    reports = gather_rank_records(local, rank, world)
    restoration = aggregate_sdpa_control_phase(
        reports,
        expected_world=world,
        variant_name=variant_name,
        phase="restoration",
    )
    lifecycle["restoration"] = restoration
    lifecycle["passed"] = lifecycle["activation"]["passed"] and restoration["passed"]
    return restoration


def begin_sdpa_arm(
    controller: SDPAArmController,
    variant: Variant,
    rank: int,
    world: int,
) -> dict:
    """Activate locally, gather all-rank evidence, and clean up failed entry."""
    error = None
    try:
        controller.activate(variant)
    except Exception as exc:
        error = exception_text(exc)
    local = {
        "phase": "activation",
        "variant": variant.name,
        "active_after_phase": controller.active,
        "error": error,
        "control": json.loads(json.dumps(controller.control, allow_nan=False))
        if controller.control is not None
        else None,
    }
    reports = gather_rank_records(local, rank, world)
    activation = aggregate_sdpa_control_phase(
        reports,
        expected_world=world,
        variant_name=variant.name,
        phase="activation",
    )
    lifecycle = {
        "variant": variant.name,
        "requested_internal_backend": variant.internal_sdpa_backend,
        "activation": activation,
        "restoration": None,
        "passed": False,
    }
    if not activation["passed"]:
        finish_sdpa_arm(controller, lifecycle, rank, world)
    return lifecycle


def final_sdpa_controller_audit(
    controller: SDPAArmController,
    rank: int,
    world: int,
) -> dict:
    """Synchronize final selector inactivity before rank zero can serialize."""
    was_active = controller.active
    error = None
    if was_active:
        try:
            controller.close()
        except Exception as exc:
            error = exception_text(exc)
    if was_active and error is None:
        error = "RuntimeError: an SDPA arm remained active until final audit"
    local = {
        "phase": "final_audit",
        "variant": "<process-finalization>",
        "had_active_context": was_active,
        "active_after_phase": controller.active,
        "error": error,
        "control": json.loads(json.dumps(controller.control, allow_nan=False))
        if controller.control is not None
        else None,
    }
    reports = gather_rank_records(local, rank, world)
    return aggregate_sdpa_control_phase(
        reports,
        expected_world=world,
        variant_name="<process-finalization>",
        phase="final_audit",
    )


def _normalized_rank_signature_summary(recorder: dict) -> list[dict]:
    summary = []
    for item in recorder["unique_signatures"]:
        signature = json.loads(json.dumps(item["signature"], allow_nan=False))
        for tensor_name in ("query", "key", "value", "attn_mask"):
            tensor = signature[tensor_name]
            if tensor is not None:
                tensor["device_index"] = None
        summary.append(
            {
                "call_count": item["call_count"],
                "signature": signature,
            }
        )
    return sorted(summary, key=_strict_json_sha256)


def aggregate_sdpa_attribution(
    rank_reports: list[dict],
    expected_world: int,
    selector_backend: str | None,
    shape_name: str,
) -> dict:
    """Require complete, identical selector evidence from every rank."""
    if expected_world < 1:
        raise ValueError("expected_world must be positive")
    ordered = sorted(rank_reports, key=lambda item: item["rank"])
    ranks = [item["rank"] for item in ordered]
    if len(set(ranks)) != len(ranks):
        raise ValueError("distributed SDPA attribution contains duplicate ranks")
    expected_ranks = list(range(expected_world))
    if ranks != expected_ranks:
        raise ValueError(
            f"distributed SDPA attribution expected ranks {expected_ranks}, got {ranks}"
        )
    if any(item["shape"] != shape_name for item in ordered):
        raise ValueError("distributed SDPA attribution mixed input-shape probes")
    if any(item["selector_backend"] != selector_backend for item in ordered):
        raise ValueError("distributed SDPA attribution mixed selector backends")

    rank_signature_summaries = [
        _normalized_rank_signature_summary(item["recorder"]) for item in ordered
    ]
    rank_signature_digests = [
        _strict_json_sha256(summary) for summary in rank_signature_summaries
    ]
    signature_agreement = len(set(rank_signature_digests)) == 1

    backend_vectors = [
        tuple(
            item["profiler"]["backend_call_counts"][name] for name in SDPA_FLAG_GETTERS
        )
        for item in ordered
    ]
    operator_count_agreement = len(set(backend_vectors)) == 1
    observed_vectors = [
        tuple(item["profiler"]["observed_backends"]) for item in ordered
    ]
    observed_backend_agreement = len(set(observed_vectors)) == 1
    selector_agreement = all(item["selector_operator_agreement"] for item in ordered)
    state_reports = [item.get("relevant_state_restore") for item in ordered]
    state_restore_agreement = all(
        state is not None and state.get("passed") for state in state_reports
    )
    state_input_vectors = [
        (
            state.get("before_trainable_state_sha256") if state else None,
            (state.get("frozen_parameters_before") or {}).get("normalized_sha256")
            if state
            else None,
            (state.get("named_buffers_before") or {}).get("normalized_sha256")
            if state
            else None,
        )
        for state in state_reports
    ]
    state_input_agreement = (
        all(None not in vector for vector in state_input_vectors)
        and len(set(state_input_vectors)) == 1
    )
    local_passed = all(item["passed"] for item in ordered)
    agreement_failing_ranks = sorted(
        {
            item["rank"]
            for index, item in enumerate(ordered)
            if (
                rank_signature_digests[index] != rank_signature_digests[0]
                or backend_vectors[index] != backend_vectors[0]
                or observed_vectors[index] != observed_vectors[0]
                or not item["selector_operator_agreement"]
                or state_input_vectors[index] != state_input_vectors[0]
                or state_reports[index] is None
                or not state_reports[index].get("passed")
            )
        }
    )

    aggregated_signatures: dict[str, dict] = {}
    for item, summary in zip(ordered, rank_signature_summaries):
        for entry in summary:
            digest = _strict_json_sha256(entry["signature"])
            aggregate = aggregated_signatures.setdefault(
                digest,
                {
                    "sha256": digest,
                    "signature": entry["signature"],
                    "total_call_count": 0,
                    "per_rank_call_count": {},
                },
            )
            aggregate["total_call_count"] += entry["call_count"]
            aggregate["per_rank_call_count"][str(item["rank"])] = entry["call_count"]

    agreement = {
        "input_signatures": signature_agreement,
        "observed_backends": observed_backend_agreement,
        "primary_operator_counts": operator_count_agreement,
        "selector_operator": selector_agreement,
        "relevant_state_inputs": state_input_agreement,
        "relevant_state_restoration": state_restore_agreement,
    }
    failing_ranks = sorted(
        {item["rank"] for item in ordered if not item["passed"]}
        | set(agreement_failing_ranks)
    )
    return {
        "shape": shape_name,
        "selector_backend": selector_backend,
        "passed": local_passed and all(agreement.values()),
        "failing_ranks": failing_ranks,
        "agreement_failing_ranks": agreement_failing_ranks,
        "all_rank_agreement": agreement,
        "rank_normalized_input_signature_sha256": [
            {"rank": item["rank"], "sha256": digest}
            for item, digest in zip(ordered, rank_signature_digests)
        ],
        "full_input_signature_aggregation": sorted(
            aggregated_signatures.values(), key=lambda item: item["sha256"]
        ),
        "rank_reports": ordered,
    }


def gather_sdpa_attribution(
    local_report: dict,
    rank: int,
    world: int,
    selector_backend: str | None,
    shape_name: str,
) -> dict:
    report = dict(local_report)
    report["rank"] = rank
    report["shape"] = shape_name
    if dist.is_initialized():
        gathered: list[dict | None] = [None] * world
        dist.all_gather_object(gathered, report)
        if any(item is None for item in gathered):
            raise RuntimeError("distributed SDPA attribution gather was incomplete")
        reports = list(gathered)
    else:
        reports = [report]
    return aggregate_sdpa_attribution(
        reports,
        expected_world=world,
        selector_backend=selector_backend,
        shape_name=shape_name,
    )


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
        if (
            requested_revision
            and len(requested_revision) == 40
            and all(
                character in "0123456789abcdefABCDEF"
                for character in requested_revision
            )
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


def validate_lora_output_head(model) -> dict:
    """Require the production LoRA profile to leave the output head untouched."""
    output_head = model.get_output_embeddings()
    if output_head is None:
        raise RuntimeError("the LoRA benchmark model exposes no output embedding head")
    adapted_names = [
        name for name, _ in output_head.named_parameters() if "lora_" in name
    ]
    trainable_names = [
        name
        for name, parameter in output_head.named_parameters()
        if parameter.requires_grad
    ]
    if adapted_names or trainable_names:
        raise RuntimeError(
            "the production LoRA benchmark requires a frozen, unadapted lm_head; "
            f"adapted={adapted_names[:5]} trainable={trainable_names[:5]}"
        )
    return {
        "frozen": True,
        "adapted": False,
        "parameter_count": sum(
            parameter.numel() for parameter in output_head.parameters()
        ),
    }


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
) -> tuple[torch.nn.Module, dict]:
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
        output_head_report = validate_lora_output_head(model)
    model.to(device)
    model.train()
    model.config.use_cache = False
    resolved_attention_backend(model, variant.attention_backend)
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
        "trainable_dtype_counts": {},
    }
    for parameter in model.parameters():
        if parameter.requires_grad:
            dtype_name = str(parameter.dtype).removeprefix("torch.")
            tuning_report["trainable_dtype_counts"][dtype_name] = (
                tuning_report["trainable_dtype_counts"].get(dtype_name, 0)
                + parameter.numel()
            )
    if tuning == "lora" and set(tuning_report["trainable_dtype_counts"]) != {"float32"}:
        raise RuntimeError(
            "the production LoRA benchmark requires FP32 trainable adapters; "
            f"found {tuning_report['trainable_dtype_counts']}"
        )
    return model, tuning_report


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


def _tensor_state_metadata(value: torch.Tensor) -> dict:
    storage = value.untyped_storage()
    return {
        "object_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device_type": value.device.type,
        "device_index": value.device.index,
        "layout": str(value.layout).removeprefix("torch."),
        "requires_grad": bool(value.requires_grad),
        "is_contiguous": bool(value.is_contiguous()),
        "is_nested": bool(getattr(value, "is_nested", False)),
        "storage_offset": int(value.storage_offset()),
        "data_ptr": int(value.data_ptr()),
        "storage_data_ptr": int(storage.data_ptr()),
        "storage_identity": int(storage._cdata),
        "storage_nbytes": int(storage.nbytes()),
    }


def _normalized_tensor_state_metadata(metadata: dict | None) -> dict | None:
    if metadata is None:
        return None
    normalized = dict(metadata)
    normalized["device_index"] = None
    normalized["data_ptr"] = None
    normalized["storage_data_ptr"] = None
    normalized["storage_identity"] = None
    return normalized


def _tensor_value_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def _qualified_registration_name(module_path: str, local_name: str) -> str:
    return f"{module_path}.{local_name}" if module_path else local_name


def _snapshot_registered_tensor_state(model, kind: str) -> dict:
    if kind not in ("frozen_parameters", "named_buffers"):
        raise ValueError(f"unknown registered tensor state kind {kind!r}")
    entries = {}
    modules = {}
    value_sha256_by_identity: dict[int, str] = {}
    buffer_values_by_identity: dict[int, torch.Tensor] = {}
    for module_path, module in unwrap(model).named_modules(remove_duplicate=False):
        registry = (
            module._parameters if kind == "frozen_parameters" else module._buffers
        )
        registration_order = []
        for local_name, value in registry.items():
            if (
                kind == "frozen_parameters"
                and value is not None
                and value.requires_grad
            ):
                continue
            registration_order.append(local_name)
            name = _qualified_registration_name(module_path, local_name)
            if name in entries:
                raise RuntimeError(f"duplicate registered tensor name {name!r}")
            value_identity = id(value) if value is not None else None
            if value is not None and value_identity not in value_sha256_by_identity:
                value_sha256_by_identity[value_identity] = _tensor_value_sha256(value)
                if kind == "named_buffers":
                    buffer_values_by_identity[value_identity] = (
                        value.detach().cpu().clone(memory_format=torch.preserve_format)
                    )
            entries[name] = {
                "name": name,
                "module_path": module_path,
                "module": module,
                "local_name": local_name,
                "object": value,
                "metadata": (
                    _tensor_state_metadata(value) if value is not None else None
                ),
                "value_sha256": (
                    value_sha256_by_identity[value_identity]
                    if value_identity is not None
                    else None
                ),
                "anchor_value": (
                    buffer_values_by_identity[value_identity]
                    if kind == "named_buffers" and value_identity is not None
                    else None
                ),
                "persistent": (
                    local_name not in module._non_persistent_buffers_set
                    if kind == "named_buffers"
                    else None
                ),
            }
        if registration_order or (
            kind == "named_buffers" and module._non_persistent_buffers_set
        ):
            modules[module_path] = {
                "module_path": module_path,
                "module": module,
                "object_type": f"{type(module).__module__}.{type(module).__qualname__}",
                "registration_order": (
                    list(registry)
                    if kind == "frozen_parameters"
                    else registration_order
                ),
                "non_persistent_buffer_names": (
                    sorted(module._non_persistent_buffers_set)
                    if kind == "named_buffers"
                    else None
                ),
            }
    return {"kind": kind, "entries": entries, "modules": modules}


def _registered_module_descriptors(state: dict) -> list[dict]:
    first_module_path: dict[int, str] = {}
    descriptors = []
    for module_path in sorted(state["modules"]):
        entry = state["modules"][module_path]
        descriptors.append(
            {
                "module_path": module_path,
                "module_alias_of": first_module_path.setdefault(
                    id(entry["module"]), module_path
                ),
                "object_type": entry["object_type"],
                "registration_order": entry["registration_order"],
                "non_persistent_buffer_names": entry["non_persistent_buffer_names"],
            }
        )
    return descriptors


def _registered_state_descriptors(state: dict, normalize_devices: bool) -> list[dict]:
    entries = state["entries"]
    first_tensor_name: dict[int, str] = {}
    first_storage_name: dict[tuple, str] = {}
    first_module_path: dict[int, str] = {}
    descriptors = []
    for name in sorted(entries):
        entry = entries[name]
        value = entry["object"]
        module = entry["module"]
        module_alias = first_module_path.setdefault(id(module), entry["module_path"])
        tensor_alias = None
        storage_alias = None
        if value is not None:
            tensor_alias = first_tensor_name.setdefault(id(value), name)
            storage_key = (
                entry["metadata"]["device_type"],
                entry["metadata"]["device_index"],
                entry["metadata"]["storage_identity"],
                entry["metadata"]["storage_nbytes"],
            )
            storage_alias = first_storage_name.setdefault(storage_key, name)
        metadata = entry["metadata"]
        if normalize_devices:
            metadata = _normalized_tensor_state_metadata(metadata)
        descriptors.append(
            {
                "name": name,
                "module_path": entry["module_path"],
                "local_name": entry["local_name"],
                "registration_is_none": value is None,
                "module_alias_of": module_alias,
                "tensor_alias_of": tensor_alias,
                "storage_alias_of": storage_alias,
                "persistent": entry["persistent"],
                "metadata": metadata,
                "value_sha256": entry["value_sha256"],
            }
        )
    return descriptors


def registered_tensor_state_report(state: dict) -> dict:
    local_descriptors = _registered_state_descriptors(state, normalize_devices=False)
    normalized_descriptors = _registered_state_descriptors(
        state, normalize_devices=True
    )
    unique_tensors = {
        id(entry["object"])
        for entry in state["entries"].values()
        if entry["object"] is not None
    }
    seen_tensors = set()
    element_count = 0
    for entry in state["entries"].values():
        value = entry["object"]
        if value is None or id(value) in seen_tensors:
            continue
        seen_tensors.add(id(value))
        element_count += value.numel()
    local_payload = {
        "modules": _registered_module_descriptors(state),
        "registrations": local_descriptors,
    }
    normalized_payload = {
        "modules": _registered_module_descriptors(state),
        "registrations": normalized_descriptors,
    }
    return {
        "kind": state["kind"],
        "sha256": _strict_json_sha256(local_payload),
        "normalized_sha256": _strict_json_sha256(normalized_payload),
        "module_registration_count": len(state["modules"]),
        "registration_count": len(state["entries"]),
        "tensor_count": len(unique_tensors),
        "none_registration_count": sum(
            entry["object"] is None for entry in state["entries"].values()
        ),
        "alias_registration_count": max(
            0,
            sum(entry["object"] is not None for entry in state["entries"].values())
            - len(unique_tensors),
        ),
        "element_count": element_count,
    }


def compare_registered_tensor_states(anchor: dict, current: dict) -> dict:
    if anchor["kind"] != current["kind"]:
        raise ValueError("registered tensor state kinds do not match")
    anchor_entries = anchor["entries"]
    current_entries = current["entries"]
    anchor_names = set(anchor_entries)
    current_names = set(current_entries)
    anchor_modules = anchor["modules"]
    current_modules = current["modules"]
    anchor_module_paths = set(anchor_modules)
    current_module_paths = set(current_modules)
    failures = {
        "missing_registrations": sorted(anchor_names - current_names),
        "extra_registrations": sorted(current_names - anchor_names),
        "module_identity_changed": [],
        "object_identity_changed": [],
        "metadata_changed": [],
        "persistence_changed": [],
        "value_changed": [],
        "missing_module_registries": sorted(anchor_module_paths - current_module_paths),
        "extra_module_registries": sorted(current_module_paths - anchor_module_paths),
        "registration_order_changed": [],
        "module_persistence_set_changed": [],
    }
    for module_path in sorted(anchor_module_paths & current_module_paths):
        before_module = anchor_modules[module_path]
        after_module = current_modules[module_path]
        display_path = module_path or "<root>"
        if before_module["module"] is not after_module["module"]:
            failures["module_identity_changed"].append(display_path)
        if before_module["registration_order"] != after_module["registration_order"]:
            failures["registration_order_changed"].append(display_path)
        if (
            before_module["non_persistent_buffer_names"]
            != after_module["non_persistent_buffer_names"]
        ):
            failures["module_persistence_set_changed"].append(display_path)
    for name in sorted(anchor_names & current_names):
        before = anchor_entries[name]
        after = current_entries[name]
        if before["module"] is not after["module"]:
            failures["module_identity_changed"].append(name)
        if before["object"] is not after["object"]:
            failures["object_identity_changed"].append(name)
        if before["metadata"] != after["metadata"]:
            failures["metadata_changed"].append(name)
        if before["persistent"] != after["persistent"]:
            failures["persistence_changed"].append(name)
        if before["value_sha256"] != after["value_sha256"]:
            failures["value_changed"].append(name)

    before_aliases = {
        item["name"]: (
            item["module_alias_of"],
            item["tensor_alias_of"],
            item["storage_alias_of"],
        )
        for item in _registered_state_descriptors(anchor, normalize_devices=False)
    }
    after_aliases = {
        item["name"]: (
            item["module_alias_of"],
            item["tensor_alias_of"],
            item["storage_alias_of"],
        )
        for item in _registered_state_descriptors(current, normalize_devices=False)
    }
    aliasing_changed = sorted(
        name
        for name in anchor_names | current_names
        if before_aliases.get(name) != after_aliases.get(name)
    )
    failures["aliasing_changed"] = aliasing_changed
    before_module_aliases = {
        item["module_path"]: item["module_alias_of"]
        for item in _registered_module_descriptors(anchor)
    }
    after_module_aliases = {
        item["module_path"]: item["module_alias_of"]
        for item in _registered_module_descriptors(current)
    }
    failures["module_aliasing_changed"] = sorted(
        path or "<root>"
        for path in anchor_module_paths | current_module_paths
        if before_module_aliases.get(path) != after_module_aliases.get(path)
    )
    failures["module_identity_changed"] = sorted(
        set(failures["module_identity_changed"])
    )
    return {
        "passed": not any(failures.values()),
        **failures,
    }


def snapshot_frozen_parameter_state(model) -> dict:
    return _snapshot_registered_tensor_state(model, "frozen_parameters")


def frozen_parameter_state_report(model_or_state) -> dict:
    state = (
        model_or_state
        if isinstance(model_or_state, dict)
        and model_or_state.get("kind") == "frozen_parameters"
        else snapshot_frozen_parameter_state(model_or_state)
    )
    return registered_tensor_state_report(state)


def snapshot_named_buffers(model) -> dict:
    return _snapshot_registered_tensor_state(model, "named_buffers")


def buffer_state_report(state: dict) -> dict:
    if state.get("kind") != "named_buffers":
        raise ValueError("expected a named-buffer state snapshot")
    return registered_tensor_state_report(state)


def restore_named_buffers(model, state: dict) -> None:
    if state.get("kind") != "named_buffers":
        raise ValueError("expected a named-buffer state snapshot")
    module_paths = dict(unwrap(model).named_modules(remove_duplicate=False))
    errors = []
    for entry in state["modules"].values():
        if module_paths.get(entry["module_path"]) is not entry["module"]:
            errors.append(f"module identity changed at {entry['module_path']!r}")
    if errors:
        raise RuntimeError("; ".join(sorted(set(errors))))

    current = snapshot_named_buffers(model)
    anchor_by_registration = {
        (id(entry["module"]), entry["local_name"]): entry
        for entry in state["entries"].values()
    }
    current_by_registration = {
        (id(entry["module"]), entry["local_name"]): entry
        for entry in current["entries"].values()
    }
    for key, entry in current_by_registration.items():
        if key in anchor_by_registration:
            continue
        entry["module"]._buffers.pop(entry["local_name"], None)
        entry["module"]._non_persistent_buffers_set.discard(entry["local_name"])

    with torch.no_grad():
        for entry in anchor_by_registration.values():
            module = entry["module"]
            local_name = entry["local_name"]
            value = entry["object"]
            module._buffers[local_name] = value
            if entry["persistent"]:
                module._non_persistent_buffers_set.discard(local_name)
            else:
                module._non_persistent_buffers_set.add(local_name)
            if value is None:
                continue
            if _tensor_state_metadata(value) != entry["metadata"]:
                errors.append(
                    f"named-buffer metadata changed irreversibly for {entry['name']}"
                )
                continue
            try:
                value.copy_(entry["anchor_value"].to(device=value.device))
            except Exception as exc:
                errors.append(
                    f"named-buffer value restore failed for {entry['name']}: "
                    f"{exception_text(exc)}"
                )
    anchor_module_ids = {id(entry["module"]) for entry in state["modules"].values()}
    for entry in current["modules"].values():
        if id(entry["module"]) in anchor_module_ids:
            continue
        entry["module"]._non_persistent_buffers_set.clear()
    restored_module_ids = set()
    for entry in state["modules"].values():
        module = entry["module"]
        if id(module) in restored_module_ids:
            continue
        restored_module_ids.add(id(module))
        ordered_buffers = {
            name: module._buffers[name] for name in entry["registration_order"]
        }
        module._buffers.clear()
        module._buffers.update(ordered_buffers)
        module._non_persistent_buffers_set.clear()
        module._non_persistent_buffers_set.update(entry["non_persistent_buffer_names"])
    if errors:
        raise RuntimeError("; ".join(sorted(set(errors))))


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
            finite_violations = difference > (atol + rtol * reference_double.abs())
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
            metrics["actual_squared_l2"] += float(actual_double.square().sum().item())
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
                summary["actual_nonzero_elements"] / summary["finite_element_count"]
                if summary["finite_element_count"]
                else None
            ),
            reference_nonzero_fraction=(
                summary["reference_nonzero_elements"] / summary["finite_element_count"]
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


def compare_tensor_maps(
    actual: dict, reference: dict, rtol: float, atol: float
) -> dict:
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
        "first_failing_gradient_tensor": gradient_parity["first_failing_tensor"],
        "worst_failing_gradient_tensor": gradient_parity["worst_failing_tensor"],
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
            actual_sensitivity["meaningful"] and reference_sensitivity["meaningful"]
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
            max_parameter_delta_relative_error=delta_parity["max_relative_error"],
            checked_parameter_delta_tensors=delta_parity["checked_tensors"],
            first_failing_parameter_delta_tensor=delta_parity["first_failing_tensor"],
            worst_failing_parameter_delta_tensor=delta_parity["worst_failing_tensor"],
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
        "gradient_nonfinite_actual_elements": gradient["nonfinite_actual_elements"],
        "gradient_nonfinite_reference_elements": gradient[
            "nonfinite_reference_elements"
        ],
        "first_failing_gradient_tensor": parity["first_failing_gradient_tensor"],
        "worst_failing_gradient_tensor": parity["worst_failing_gradient_tensor"],
        "parameter_delta_status": parity["parameter_delta_status"],
        "parameter_delta_max_absolute_error": parity["max_parameter_delta_abs_error"],
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
            if item["loss_nonfinite_actual"] or item["loss_nonfinite_reference"]
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
        "parameter_delta_status": _aggregate_status(ordered, "parameter_delta_status"),
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
    parity["loss_nonfinite_actual"] = bool(distributed["loss_nonfinite_actual_ranks"])
    parity["loss_nonfinite_reference"] = bool(
        distributed["loss_nonfinite_reference_ranks"]
    )
    parity["nonfinite_loss_ranks"] = distributed["nonfinite_loss_ranks"]
    parity["max_gradient_abs_error"] = distributed["gradient_max_absolute_error_max"]
    parity["max_gradient_relative_error"] = distributed[
        "gradient_max_relative_error_max"
    ]
    parity["gradient_relative_l2_error"] = distributed["gradient_relative_l2_error_max"]
    parity["gradient_cosine_similarity"] = distributed["gradient_cosine_similarity_min"]
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
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
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
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
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


def profile_sdpa_attribution_probe(
    model,
    variant: Variant,
    input_ids: torch.Tensor,
    weights: torch.Tensor,
    rank: int,
    world: int,
    device: torch.device,
    anchor_state: dict[str, torch.Tensor],
    anchor_digest: str,
    seed: int,
    shape_name: str,
) -> dict:
    """Profile one rank-local, grad-enabled forward and restore relevant state.

    The caught region intentionally contains no distributed collective, DDP
    forward, backward pass, or optimizer step. Every rank reaches the one final
    evidence gather even when another rank's local forward fails.
    """
    recorder = SDPAInputRecorder()
    profiler_result = parse_sdpa_profiler_events([])
    errors: list[str] = []
    before_digest = None
    after_digest = None
    frozen_anchor = None
    frozen_before = None
    frozen_after_state = None
    frozen_after = None
    frozen_comparison = None
    buffer_anchor = None
    buffer_before = None
    buffer_post_forward_state = None
    buffer_post_forward = None
    buffer_post_forward_comparison = None
    buffer_after_restore_state = None
    buffer_after_restore = None
    buffer_after_restore_comparison = None
    cpu_rng_state = None
    cuda_rng_state = None
    rng_restored = False

    try:
        restore_trainable_state(model, anchor_state)
        before_digest = trainable_state_digest(model)
        if before_digest != anchor_digest:
            raise RuntimeError(
                "the attribution probe did not start from the exact warmed anchor"
            )
        frozen_anchor = snapshot_frozen_parameter_state(model)
        frozen_before = frozen_parameter_state_report(frozen_anchor)
        buffer_anchor = snapshot_named_buffers(model)
        buffer_before = buffer_state_report(buffer_anchor)
        cpu_rng_state = torch.random.get_rng_state().clone()
        cuda_rng_state = torch.cuda.get_rng_state(device).clone()
        model.zero_grad(set_to_none=True)
        torch.random.default_generator.manual_seed(seed + rank)
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            with recorder:
                with torch.enable_grad():
                    probe_output = forward_sum(
                        unwrap(model), variant, input_ids, weights
                    )
                del probe_output
            torch.cuda.synchronize(device)
        profiler_result = parse_sdpa_profiler_events(profiler.key_averages())
    except Exception as exc:
        errors.append(exception_text(exc))
    finally:
        try:
            frozen_after_state = snapshot_frozen_parameter_state(model)
            frozen_after = frozen_parameter_state_report(frozen_after_state)
            if frozen_anchor is not None:
                frozen_comparison = compare_registered_tensor_states(
                    frozen_anchor, frozen_after_state
                )
        except Exception as exc:
            errors.append(
                f"frozen-parameter verification failed: {exception_text(exc)}"
            )
        if buffer_anchor is not None:
            try:
                buffer_post_forward_state = snapshot_named_buffers(model)
                buffer_post_forward = buffer_state_report(buffer_post_forward_state)
                buffer_post_forward_comparison = compare_registered_tensor_states(
                    buffer_anchor, buffer_post_forward_state
                )
            except Exception as exc:
                errors.append(f"named-buffer snapshot failed: {exception_text(exc)}")
        try:
            model.zero_grad(set_to_none=True)
            restore_trainable_state(model, anchor_state)
        except Exception as exc:
            errors.append(f"trainable-state restore failed: {exception_text(exc)}")
        if buffer_anchor is not None:
            try:
                restore_named_buffers(model, buffer_anchor)
            except Exception as exc:
                errors.append(f"named-buffer restore failed: {exception_text(exc)}")
        if cpu_rng_state is not None:
            try:
                torch.random.set_rng_state(cpu_rng_state)
            except Exception as exc:
                errors.append(f"CPU RNG restore failed: {exception_text(exc)}")
        if cuda_rng_state is not None:
            try:
                torch.cuda.set_rng_state(cuda_rng_state, device)
            except Exception as exc:
                errors.append(f"CUDA RNG restore failed: {exception_text(exc)}")
        try:
            torch.cuda.synchronize(device)
        except Exception as exc:
            errors.append(f"post-restore synchronization failed: {exception_text(exc)}")

    try:
        after_digest = trainable_state_digest(model)
    except Exception as exc:
        errors.append(f"trainable-state verification failed: {exception_text(exc)}")
    try:
        buffer_after_restore_state = snapshot_named_buffers(model)
        buffer_after_restore = buffer_state_report(buffer_after_restore_state)
        if buffer_anchor is not None:
            buffer_after_restore_comparison = compare_registered_tensor_states(
                buffer_anchor, buffer_after_restore_state
            )
    except Exception as exc:
        errors.append(f"named-buffer verification failed: {exception_text(exc)}")
    if cpu_rng_state is not None and cuda_rng_state is not None:
        try:
            rng_restored = torch.equal(
                torch.random.get_rng_state(), cpu_rng_state
            ) and torch.equal(torch.cuda.get_rng_state(device), cuda_rng_state)
        except Exception as exc:
            errors.append(f"RNG verification failed: {exception_text(exc)}")

    trainable_restored = (
        before_digest == anchor_digest and after_digest == anchor_digest
    )
    frozen_unchanged = bool(
        frozen_comparison is not None and frozen_comparison["passed"]
    )
    buffers_restored = bool(
        buffer_after_restore_comparison is not None
        and buffer_after_restore_comparison["passed"]
    )
    if not trainable_restored:
        errors.append("attribution trainable state was not restored exactly")
    if not frozen_unchanged:
        errors.append("attribution mutated a frozen parameter")
    if not buffers_restored:
        errors.append("attribution named buffers were not restored exactly")
    if not rng_restored:
        errors.append("attribution RNG state was not restored exactly")

    relevant_state_restore = {
        "passed": (
            trainable_restored
            and frozen_unchanged
            and buffers_restored
            and rng_restored
        ),
        "before_trainable_state_sha256": before_digest,
        "after_trainable_state_sha256": after_digest,
        "expected_trainable_state_sha256": anchor_digest,
        "frozen_parameters_before": frozen_before,
        "frozen_parameters_after": frozen_after,
        "frozen_parameter_comparison": frozen_comparison,
        "frozen_parameters_unchanged": frozen_unchanged,
        "named_buffers_before": buffer_before,
        "named_buffers_post_forward": buffer_post_forward,
        "named_buffers_after_restore": buffer_after_restore,
        "named_buffers_post_forward_comparison": (buffer_post_forward_comparison),
        "named_buffers_after_restore_comparison": (buffer_after_restore_comparison),
        "named_buffers_mutated_during_probe": (
            buffer_post_forward_comparison is None
            or not buffer_post_forward_comparison["passed"]
        ),
        "named_buffers_restored_exactly": buffers_restored,
        "rng_restored_exactly": rng_restored,
        "scope": {
            "trainable_parameters": "exact bytes restored and verified",
            "frozen_parameters": (
                "registration, identity, aliasing, metadata, and exact bytes "
                "verified unchanged"
            ),
            "named_buffers": (
                "registration, identity, aliasing, persistence, metadata, and "
                "exact bytes restored and verified"
            ),
            "cpu_rng": "default generator restored exactly",
            "local_cuda_rng": "default generator restored exactly",
            "unregistered_python_state": (
                "out of scope; supported models must not mutate unregistered "
                "Python-side caches when use_cache=False"
            ),
        },
    }
    try:
        recorder_report = recorder.report()
    except Exception as exc:
        errors.append(f"SDPA recorder reporting failed: {exception_text(exc)}")
        recorder_report = {
            "total_calls": 0,
            "unique_signature_count": 0,
            "ordered_signature_sha256": _strict_json_sha256([]),
            "unique_signatures": [],
        }
    try:
        local = evaluate_local_sdpa_attribution(
            variant.internal_sdpa_backend,
            recorder_report,
            profiler_result,
            error="; ".join(errors) if errors else None,
        )
    except Exception as exc:
        local = evaluate_local_sdpa_attribution(
            variant.internal_sdpa_backend,
            {
                "total_calls": 0,
                "unique_signature_count": 0,
                "ordered_signature_sha256": _strict_json_sha256([]),
                "unique_signatures": [],
            },
            parse_sdpa_profiler_events([]),
            error=(
                "; ".join(
                    errors
                    + [f"SDPA attribution evaluation failed: {exception_text(exc)}"]
                )
            ),
        )
    local["relevant_state_restore"] = relevant_state_restore
    return gather_sdpa_attribution(
        local,
        rank,
        world,
        variant.internal_sdpa_backend,
        shape_name,
    )


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
    training_step(model, optimizer, variant, input_ids, weights, tuning, world, device)
    torch.cuda.synchronize(device)
    first_post_attribution_training_step_seconds = distributed_max(
        time.perf_counter() - started, device
    )

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
        "first_post_attribution_training_step_seconds": (
            first_post_attribution_training_step_seconds
        ),
        "p50_step_seconds": percentile(durations, 0.50),
        "p95_step_seconds": percentile(durations, 0.95),
        "mean_step_seconds": statistics.fmean(durations),
        "raw_tokens_per_step": raw_tokens_per_step,
        "target_tokens_per_step": target_tokens_per_step,
        "raw_tokens_per_second": raw_tokens_per_step
        * measured_steps
        / measured_seconds,
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
        default="",
        help=("optional candidate name; the selected reference is added automatically"),
    )
    parser.add_argument(
        "--reference-variant",
        choices=list(VARIANTS_BY_NAME),
        default=REFERENCE_VARIANT.name,
        help=(
            "parity anchor and standalone arm when --variants is empty; forced "
            "internal SDPA backends can therefore run without the automatic arm"
        ),
    )
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--tuning",
        choices=["lora", "full"],
        default="lora",
        help="train FP32 LoRA adapters by default; full is an explicit separate profile",
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
    parser.add_argument(
        "--output", type=Path, default=Path("a100-kernel-benchmark.json")
    )
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
        raise ValueError(
            "--learning-rate must be positive and --weight-decay nonnegative"
        )
    if args.lora_r < 1 or args.lora_alpha < 1:
        raise ValueError("--lora-r and --lora-alpha must be positive")
    if (
        min(
            args.parity_rtol,
            args.parity_atol,
            args.parameter_delta_rtol,
            args.parameter_delta_atol,
        )
        < 0
    ):
        raise ValueError("parity tolerances must be nonnegative")


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
        raise ValueError("YETO_GIT_SHA and YETO_GIT_DIRTY must be supplied together")
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
        report["completed_variants"] == report["planned_variants"]
        and len(set(report["completed_variants"])) == len(report["completed_variants"])
        and all(record["status"] == "passed" for record in report["variants"])
    ):
        report["status"] = "passed"
        report["fatal"] = None
    else:
        report["status"] = "incomplete"
        report["fatal"] = None


def write_report_atomic(output: Path, serialized: str) -> None:
    """Durably replace one report without exposing a partial JSON artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with stream:
            stream.write(serialized)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def publish_report_collectively(
    report: dict,
    output: Path,
    rank: int,
    failed: bool,
    fatal_phase: str | None,
    fatal_reason: str | None,
) -> dict:
    """Publish on rank zero, then give every rank the same success decision."""
    result = None
    serialized = None
    if rank == 0:
        try:
            finalize_report_status(report, failed, fatal_phase, fatal_reason)
            serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
            write_report_atomic(output, serialized)
            result = {
                "passed": True,
                "error": None,
                "output": str(output),
                "status": report["status"],
            }
        except Exception as exc:
            result = {
                "passed": False,
                "error": exception_text(exc),
                "output": str(output),
                "status": "failed",
            }
    result = broadcast_object(result, rank)
    if rank == 0:
        try:
            if result["passed"]:
                print(serialized, flush=True)
                print(f"wrote {output}", file=sys.stderr, flush=True)
            else:
                print(
                    f"report publication failed: {result['error']}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception:
            # The atomic report and collective decision are authoritative; a
            # closed diagnostic stream must not strand other ranks.
            pass
    return result


def restore_sdpa_arm_for_record(
    record: dict,
    controller: SDPAArmController,
    lifecycle: dict,
    rank: int,
    world: int,
) -> str | None:
    """Attach all-rank restoration evidence and invalidate leaked-arm timing."""
    restoration = finish_sdpa_arm(controller, lifecycle, rank, world)
    record["sdpa_backend_control"] = lifecycle
    if restoration["passed"]:
        return None
    reason = f"SDPA selector restoration failed on ranks {restoration['failing_ranks']}"
    previous = record.get("reason")
    record["reason"] = f"{previous}; {reason}" if previous else reason
    record["status"] = "failed"
    record.pop("metrics", None)
    return reason


def _main(argv, backend_controller: SDPAArmController) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    reference_variant = VARIANTS_BY_NAME[args.reference_variant]
    variants = select_variants(args.variants, reference_variant.name)
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
        "schema_version": 4,
        "benchmark": "a100-causal-training-kernels",
        "status": "incomplete",
        "reference_variant": reference_variant.name,
        "planned_variants": [variant.name for variant in variants],
        "completed_variants": [],
        "fatal": None,
        "trial": {
            "index": args.trial_index,
            "timing_trials_in_record": 1,
            "aggregation": "aggregate separate JSON records by trial.index",
        },
        "sdpa_backend_attribution_contract": {
            "selector_api": "torch.nn.attention.sdpa_kernel",
            "capability_api": "torch.backends.cuda.SDPAParams and can_use_*_attention",
            "operator_evidence": "torch.profiler",
            "forward_only_aten_allowlist": sorted(SDPA_FORWARD_ATEN_ALLOWLIST),
            "unexpected_aten_attention_operator_is_fatal": True,
            "required_shapes": ["parity", "timing"],
            "probe_execution": "rank-local grad-enabled forward only",
            "automatic_backend_policy": (
                "exactly one recognized observed backend per probe"
            ),
            "timing_requires_attribution_pass": True,
            "timing_shape_is_profiled_before_timing": True,
            "first_timing_metric": ("first_post_attribution_training_step_seconds"),
            "selector_lifecycle_requires_all_ranks": True,
            "registered_state_scope": (
                "registration, identity, aliasing, persistence, metadata, bytes"
            ),
            "report_publication": "strict JSON, atomic replace, all-rank decision",
            "reference_variant": reference_variant.name,
            "publishable_variant_limit": (
                "selected reference plus at most one candidate"
            ),
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
        backend_lifecycle = begin_sdpa_arm(backend_controller, variant, rank, world)
        if not backend_lifecycle["activation"]["passed"]:
            activation_reason = (
                "SDPA selector activation failed on ranks "
                f"{backend_lifecycle['activation']['failing_ranks']}"
            )
            record = {
                "variant": asdict(variant),
                "sdpa_backend_control": backend_lifecycle,
                "status": "failed",
                "reason": activation_reason,
            }
            if rank == 0:
                report["variants"].append(record)
            failed = True
            fatal_phase = "sdpa_selector_activation"
            fatal_reason = activation_reason
            break
        setup_started = time.perf_counter()
        model = None
        tuning_report = None
        error = None
        fatal_load_error = False
        try:
            torch.manual_seed(model_init_seed)
            torch.cuda.manual_seed_all(model_init_seed)
            model, tuning_report = load_raw_model(
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
            load_reason = errors[0] if errors else "model load failed on another rank"
            record = {
                "variant": asdict(variant),
                "status": (
                    "failed"
                    if variant == reference_variant or fatal_load_error
                    else "skipped"
                ),
                "reason": load_reason,
            }
            restoration_reason = restore_sdpa_arm_for_record(
                record,
                backend_controller,
                backend_lifecycle,
                rank,
                world,
            )
            if rank == 0:
                report["variants"].append(record)
            if restoration_reason is not None:
                failed = True
                fatal_phase = "sdpa_selector_restoration"
                fatal_reason = restoration_reason
                break
            if variant == reference_variant or fatal_load_error:
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
                "constructed trainable state did not match the selected "
                "reference; check model/adapter initialization seeding"
            )
            record = {
                "variant": asdict(variant),
                "status": "failed",
                "reason": state_reason,
                "tuning": tuning_report,
                "construction_trainable_state_sha256": construction_digest,
                "reference_construction_trainable_state_sha256": (
                    construction_reference_digest
                ),
                "state_digest_reports": {
                    "construction": construction_state_report,
                },
            }
            del model
            cleanup_cuda()
            restoration_reason = restore_sdpa_arm_for_record(
                record,
                backend_controller,
                backend_lifecycle,
                rank,
                world,
            )
            if rank == 0:
                report["variants"].append(record)
            failed = True
            fatal_phase = (
                "sdpa_selector_restoration"
                if restoration_reason is not None
                else "construction_state_validation"
            )
            fatal_reason = restoration_reason or state_reason
            break

        vocab_size = int(unwrap(model).config.vocab_size)
        controlled_warmup_report = None
        if parity_anchor_state is None:
            if variant != reference_variant:
                raise RuntimeError(
                    "the first benchmark variant must establish the anchor"
                )
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
                    reference_variant,
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
                    "source_variant": reference_variant.name,
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
                "restored trainable parity anchor did not match the selected "
                "reference's warmed state"
            )
            record = {
                "variant": asdict(variant),
                "status": "failed",
                "reason": anchor_reason,
                "tuning": tuning_report,
                "restored_trainable_state_sha256": restored_anchor_digest,
                "parity_anchor_trainable_state_sha256": parity_anchor_digest,
                "state_digest_reports": {
                    "construction": construction_state_report,
                    "warmed_anchor_validation": anchor_state_report,
                },
            }
            del model
            cleanup_cuda()
            restoration_reason = restore_sdpa_arm_for_record(
                record,
                backend_controller,
                backend_lifecycle,
                rank,
                world,
            )
            if rank == 0:
                report["variants"].append(record)
            failed = True
            fatal_phase = (
                "sdpa_selector_restoration"
                if restoration_reason is not None
                else "parity_anchor_restore"
            )
            fatal_reason = restoration_reason or anchor_reason
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
            parameter_delta_witness(model, optimizer, device, restore_parameters=True)
        )
        del optimizer
        repeat_seconds = distributed_max(time.perf_counter() - repeat_started, device)
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
                    f"self-repeat control failed: {self_repeat_parity['reason']}"
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
                set(parity["overall_failing_ranks"]) | set(timing_anchor_failing_ranks)
            )

        parity_seconds = distributed_max(time.perf_counter() - parity_started, device)
        parity_passed = parity["passed"]
        parity["seconds"] = parity_seconds
        parity["first_forward_backward_compile_seconds"] = compile_seconds
        parity["optimizer_init_step_seconds"] = optimizer_init_seconds
        parity["first_backward"] = backward_report
        parity["self_repeat_seconds"] = repeat_seconds
        parity["self_repeat_forward_backward_seconds"] = repeat_forward_backward_seconds
        parity["self_repeat_optimizer_step_seconds"] = repeat_optimizer_step_seconds
        parity["self_repeat_backward"] = repeat_backward_report
        parity["timing_anchor_restored_all_ranks"] = timing_anchor_matches_all_ranks
        parity["timing_anchor_restore_failing_ranks"] = timing_anchor_failing_ranks
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
            del model
            cleanup_cuda()
            restoration_reason = restore_sdpa_arm_for_record(
                record,
                backend_controller,
                backend_lifecycle,
                rank,
                world,
            )
            if rank == 0:
                report["variants"].append(record)
            failed = True
            fatal_phase = (
                "sdpa_selector_restoration"
                if restoration_reason is not None
                else "parity"
            )
            fatal_reason = restoration_reason or parity["reason"]
            break

        if not is_reference:
            # The full reference snapshots stay resident for later variants;
            # a passing candidate snapshot is no longer needed during timing.
            del signature
            del parameter_deltas
        del repeat_signature
        del repeat_parameter_deltas

        input_ids, weights = make_batch(
            vocab_size,
            args.micro_batch_size,
            args.seq_len,
            args.target_fraction,
            args.seed + 10_000 + rank,
            device,
        )
        attribution_started = time.perf_counter()
        parity_shape_attribution = profile_sdpa_attribution_probe(
            model,
            variant,
            parity_ids,
            parity_weights,
            rank,
            world,
            device,
            parity_anchor_state,
            parity_anchor_digest,
            args.seed + 70_000,
            "parity",
        )
        timing_shape_attribution = profile_sdpa_attribution_probe(
            model,
            variant,
            input_ids,
            weights,
            rank,
            world,
            device,
            parity_anchor_state,
            parity_anchor_digest,
            args.seed + 80_000,
            "timing",
        )
        restore_trainable_state(model, parity_anchor_state)
        torch.cuda.synchronize(device)
        post_attribution_digest = trainable_state_digest(model)
        post_attribution_state_report = gather_state_digest_diagnostics(
            post_attribution_digest,
            rank,
            world,
            reference_digest=parity_anchor_digest,
        )
        attribution_seconds = distributed_max(
            time.perf_counter() - attribution_started, device
        )
        attribution = {
            "passed": (
                parity_shape_attribution["passed"]
                and timing_shape_attribution["passed"]
                and post_attribution_state_report["passed"]
            ),
            "seconds_not_in_timing_metrics": attribution_seconds,
            "parity_shape": parity_shape_attribution,
            "timing_shape": timing_shape_attribution,
            "post_attribution_anchor_restore": post_attribution_state_report,
            "post_attribution_trainable_state_sha256": post_attribution_digest,
        }
        parity["state_digest_reports"]["post_attribution_restore"] = (
            post_attribution_state_report
        )
        if not attribution["passed"]:
            failed_shapes = [
                name
                for name, result in (
                    ("parity", parity_shape_attribution),
                    ("timing", timing_shape_attribution),
                )
                if not result["passed"]
            ]
            if not post_attribution_state_report["passed"]:
                failed_shapes.append("anchor_restore")
            attribution_reason = (
                "pre-timing SDPA backend attribution failed for "
                + ", ".join(failed_shapes)
            )
            record = {
                "variant": asdict(variant),
                "status": "failed",
                "reason": attribution_reason,
                "setup_seconds": setup_seconds,
                "tuning": tuning_report,
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
                "sdpa_backend_attribution": attribution,
            }
            del model
            cleanup_cuda()
            restoration_reason = restore_sdpa_arm_for_record(
                record,
                backend_controller,
                backend_lifecycle,
                rank,
                world,
            )
            if rank == 0:
                report["variants"].append(record)
            failed = True
            fatal_phase = (
                "sdpa_selector_restoration"
                if restoration_reason is not None
                else "sdpa_backend_attribution"
            )
            fatal_reason = restoration_reason or attribution_reason
            break

        optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
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
        record = {
            "variant": asdict(variant),
            "status": "passed",
            "setup_seconds": setup_seconds,
            "tuning": tuning_report,
            "model_init_seed": model_init_seed,
            "adapter_init_seed": (adapter_init_seed if args.tuning == "lora" else None),
            "construction_trainable_state_sha256": construction_digest,
            "reference_construction_trainable_state_sha256": (
                construction_reference_digest
            ),
            "parity_anchor_trainable_state_sha256": parity_anchor_digest,
            "state_digest_reports": parity["state_digest_reports"],
            "parity": parity,
            "sdpa_backend_attribution": attribution,
            "metrics": metrics,
        }
        del optimizer
        del model
        cleanup_cuda()
        restoration_reason = restore_sdpa_arm_for_record(
            record,
            backend_controller,
            backend_lifecycle,
            rank,
            world,
        )
        if rank == 0:
            report["variants"].append(record)
        if restoration_reason is not None:
            failed = True
            fatal_phase = "sdpa_selector_restoration"
            fatal_reason = restoration_reason
            break

    final_selector_audit = final_sdpa_controller_audit(backend_controller, rank, world)
    report["sdpa_selector_finalization"] = final_selector_audit
    if not final_selector_audit["passed"]:
        failed = True
        fatal_phase = "sdpa_selector_finalization"
        fatal_reason = (
            "final SDPA selector audit failed on ranks "
            f"{final_selector_audit['failing_ranks']}"
        )
    publication = publish_report_collectively(
        report,
        args.output,
        rank,
        failed,
        fatal_phase,
        fatal_reason,
    )
    if not publication["passed"]:
        failed = True
    return 1 if failed else 0


def main(argv=None) -> int:
    backend_controller = SDPAArmController()
    active_exception = (None, None, None)
    try:
        return _main(argv, backend_controller)
    except BaseException:
        active_exception = sys.exc_info()
        raise
    finally:
        try:
            backend_controller.close(active_exception)
        finally:
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    if active_exception[0] is None:
                        raise


if __name__ == "__main__":
    raise SystemExit(main())
