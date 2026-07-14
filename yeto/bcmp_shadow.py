"""Behavior-preserving AdamW state diagnostics at fragment broadcasts.

The learner overwrites (or blends) a fragment's parameters after an optimizer
step while leaving AdamW's moments untouched.  This module computes three
counterfactual first-moment transports at the *first clipped gradient after the
broadcast* without mutating parameters, gradients, or optimizer state:

``ray`` (BC-MP)
    Project the bias-corrected old first moment onto ``a * g``, with
    ``a = clip(<g, P m>/<g, P g>, 0, 1)``.

``slab`` (conservative alternative)
    Make the minimum change parallel to ``g`` needed to put the old moment's
    preconditioned work in ``[0, <g, P g>]``.  This preserves the entire
    P-orthogonal component and is therefore less destructive, but it has no
    finite-loss safety guarantee.

``reset`` (hard-reset control)
    Replace the raw first moment by zero.  This deliberately blunt control
    distinguishes the value of BC-MP's gradient-aligned carry from merely
    deleting all pre-broadcast first-moment history.

``P`` is the exact positive diagonal denominator that the upcoming AdamW step
will use, including the would-be second-moment update and bias correction.  The
returned step tensors are positive displacements (AdamW subtracts them).

This is deliberately an evidence tool, not an optimizer implementation.  A
caller may retain the candidate-minus-stock directions until the next clipped
gradient and score predictive alignment with :func:`score_directions`.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .fragments import Fragment


SHADOW_SCHEMA = "bcmp_shadow_v1"
RESOLUTION_SCHEMA = "bcmp_shadow_resolution_v1"
DROP_SCHEMA = "bcmp_shadow_drop_v1"
RUN_SCHEMA = "bcmp_shadow_run_v1"
RUN_SUMMARY_SCHEMA = "bcmp_shadow_run_summary_v1"


@dataclass
class BCMPShadowTensors:
    """Optional counterfactual tensors retained for tests/future resolution.

    All mappings use the fragment's parameter names.  Step tensors are the
    total AdamW displacement, including the common decoupled weight-decay
    displacement.  Raw moments are values that would replace ``exp_avg``
    immediately before the factual ``optimizer.step()``.
    """

    stock_step: dict[str, torch.Tensor]
    ray_step: dict[str, torch.Tensor]
    slab_step: dict[str, torch.Tensor]
    reset_step: dict[str, torch.Tensor]
    ray_minus_stock: dict[str, torch.Tensor]
    slab_minus_stock: dict[str, torch.Tensor]
    reset_minus_stock: dict[str, torch.Tensor]
    ray_raw_exp_avg: dict[str, torch.Tensor]
    slab_raw_exp_avg: dict[str, torch.Tensor]
    reset_raw_exp_avg: dict[str, torch.Tensor]


@dataclass
class BCMPShadowResult:
    """JSON-safe scalar record plus optional device-resident tensors."""

    record: dict[str, Any]
    tensors: BCMPShadowTensors | None = None


@dataclass
class _Entry:
    name: str
    param: torch.Tensor
    grad: torch.Tensor
    raw_m: torch.Tensor
    m_hat: torch.Tensor
    preconditioner: torch.Tensor
    beta1: float
    beta1_power: float
    step: int
    lr: float
    weight_decay: float


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"AdamW {field} must be scalar, got shape {tuple(value.shape)}")
        value = value.detach().item()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"AdamW {field} must be finite, got {result}")
    return result


def _state_step(state: dict[str, Any]) -> int:
    if not state:
        return 0
    raw = state.get("step", 0)
    value = _as_float(raw, "state step")
    rounded = int(round(value))
    if value < 0 or abs(value - rounded) > 1e-6:
        raise ValueError(f"AdamW state step must be a non-negative integer, got {value}")
    return rounded


def _safe_sqrt(value: torch.Tensor) -> float:
    scalar = float(value.detach().item())
    # Reduction roundoff can make a theoretically non-negative norm square a
    # few ulps negative.  A material negative value is an implementation bug.
    if scalar < 0.0:
        tol = 64.0 * torch.finfo(value.dtype).eps * max(1.0, abs(scalar))
        if scalar < -tol:
            raise RuntimeError(f"negative accumulated norm square: {scalar}")
        scalar = 0.0
    return math.sqrt(scalar)


def _json_float(value: torch.Tensor | float) -> float:
    scalar = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"non-finite BC-MP diagnostic scalar: {scalar}")
    return scalar


def _optimizer_groups(optimizer: torch.optim.Optimizer) -> dict[int, dict[str, Any]]:
    by_param: dict[int, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            key = id(param)
            if key in by_param:
                raise ValueError("a parameter occurs in more than one optimizer group")
            by_param[key] = group
    return by_param


def broadcast_jump_stats(
    fragment: Fragment,
    params: dict[str, torch.Tensor],
    global_flat: torch.Tensor,
    merge_alpha: float,
) -> dict[str, float]:
    """Return the actual pre-apply broadcast jump without changing parameters.

    The learner applies ``alpha * local + (1-alpha) * global``.  This helper is
    intended to run only for sampled shadow events because it materializes one
    fragment-sized flat tensor and synchronizes when returning JSON scalars.
    """
    if not 0.0 <= merge_alpha < 1.0:
        raise ValueError(f"merge_alpha must be in [0, 1), got {merge_alpha}")
    if global_flat.numel() != fragment.numel:
        raise ValueError(
            f"global fragment has {global_flat.numel()} values, expected {fragment.numel}"
        )
    local = torch.cat(
        [params[name].detach().reshape(-1).float() for name, _ in fragment.tensors]
    )
    global_flat = global_flat.detach().to(device=local.device, dtype=local.dtype)
    jump = (1.0 - merge_alpha) * (global_flat - local)
    jump_l2 = float(torch.linalg.vector_norm(jump).item())
    local_l2 = float(torch.linalg.vector_norm(local).item())
    return {
        "broadcast_jump_l2": jump_l2,
        "broadcast_local_l2": local_l2,
        "broadcast_jump_relative_l2": jump_l2 / max(local_l2, torch.finfo(local.dtype).tiny),
    }


def compute_bcmp_shadow(
    fragment: Fragment,
    params: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    fragment_id: int,
    broadcast_version: int,
    broadcast_local_step: int,
    gradient_local_step: int,
    learner_id: int,
    rank: int,
    jump_stats: dict[str, float] | None = None,
    capture_tensors: bool | str = False,
    accum_dtype: torch.dtype = torch.float64,
) -> BCMPShadowResult:
    """Compute stock/ray/slab next-step counterfactuals without mutation.

    The caller must invoke this after gradient all-reduce and clipping and
    before the factual AdamW step.  ``gradient_local_step`` is the learner's
    completed-step counter at that point; the upcoming optimizer step is
    therefore ``gradient_local_step + 1`` in human one-based notation.
    """
    if not fragment.tensors:
        raise ValueError("BC-MP shadow requires a non-empty fragment")
    if accum_dtype not in (torch.float32, torch.float64):
        raise ValueError("accum_dtype must be torch.float32 or torch.float64")
    if capture_tensors not in (False, True, "directions"):
        raise ValueError("capture_tensors must be false, true, or 'directions'")
    capture_mode = (
        "all" if capture_tensors is True
        else "directions" if capture_tensors == "directions"
        else None
    )

    group_for = _optimizer_groups(optimizer)
    first_param = params[fragment.tensors[0][0]]
    device = first_param.device
    zero = torch.zeros((), dtype=accum_dtype, device=device)
    numerator = zero.clone()
    denominator = zero.clone()
    old_pnorm2 = zero.clone()
    grad_norm2 = zero.clone()
    param_norm2 = zero.clone()
    entries: list[_Entry] = []
    group_ids: set[int] = set()
    steps: list[int] = []

    for name, expected_numel in fragment.tensors:
        if name not in params:
            raise KeyError(f"fragment parameter {name!r} is missing")
        param = params[name]
        if param.device != device:
            raise ValueError("all parameters in a fragment must be on one device")
        if param.numel() != expected_numel:
            raise ValueError(
                f"fragment records {expected_numel} values for {name}, got {param.numel()}"
            )
        if param.is_complex():
            raise ValueError("BC-MP shadow does not support complex parameters")
        if param.grad is None:
            raise ValueError(f"fragment parameter {name!r} has no post-clip gradient")
        if id(param) not in group_for:
            raise ValueError(f"fragment parameter {name!r} is not in the optimizer")

        group = group_for[id(param)]
        group_ids.add(id(group))
        beta1, beta2 = map(float, group["betas"])
        eps = _as_float(group["eps"], "epsilon")
        lr = _as_float(group["lr"], "learning rate")
        weight_decay = _as_float(group["weight_decay"], "weight decay")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"AdamW betas must lie in [0,1), got {(beta1, beta2)}")
        if eps < 0.0 or lr < 0.0 or weight_decay < 0.0:
            raise ValueError("AdamW epsilon, learning rate, and weight decay must be non-negative")

        state = optimizer.state.get(param, {})
        step = _state_step(state)
        steps.append(step)
        g = param.grad.detach().to(dtype=accum_dtype)
        if bool(group.get("maximize", False)):
            g = -g
        if state:
            try:
                raw_m = state["exp_avg"].detach().to(dtype=accum_dtype)
                raw_v = state["exp_avg_sq"].detach().to(dtype=accum_dtype)
            except KeyError as exc:
                raise ValueError(f"incomplete AdamW state for {name}: missing {exc.args[0]}") from exc
        else:
            raw_m = torch.zeros_like(g)
            raw_v = torch.zeros_like(g)
        if raw_m.shape != param.shape or raw_v.shape != param.shape:
            raise ValueError(f"AdamW moment shape mismatch for {name}")

        beta1_power = beta1**step
        bc1 = 1.0 - beta1_power
        m_hat = raw_m / bc1 if step > 0 else torch.zeros_like(raw_m)
        v_next = beta2 * raw_v + (1.0 - beta2) * g.square()
        if bool(group.get("amsgrad", False)):
            if state and "max_exp_avg_sq" not in state:
                raise ValueError(f"AMSGrad state for {name} lacks max_exp_avg_sq")
            old_max = (
                state["max_exp_avg_sq"].detach().to(dtype=accum_dtype)
                if state
                else torch.zeros_like(v_next)
            )
            v_next = torch.maximum(old_max, v_next)
        bc2_next = 1.0 - beta2 ** (step + 1)
        pdiag = 1.0 / (torch.sqrt(v_next / bc2_next) + eps)

        numerator += torch.sum(g * pdiag * m_hat)
        denominator += torch.sum(g * pdiag * g)
        old_pnorm2 += torch.sum(m_hat * pdiag * m_hat)
        grad_norm2 += torch.sum(g.square())
        param_norm2 += torch.sum(param.detach().to(dtype=accum_dtype).square())
        entries.append(
            _Entry(
                name=name,
                param=param,
                grad=g,
                raw_m=raw_m,
                m_hat=m_hat,
                preconditioner=pdiag,
                beta1=beta1,
                beta1_power=beta1_power,
                step=step,
                lr=lr,
                weight_decay=weight_decay,
            )
        )

    n_value = _json_float(numerator)
    d_value = _json_float(denominator)
    fallback_reason: str | None = None
    if d_value > 0.0:
        a_raw_tensor = numerator / denominator
        a_tensor = torch.clamp(a_raw_tensor, 0.0, 1.0)
        a_raw = _json_float(a_raw_tensor)
        a = _json_float(a_tensor)
    else:
        # A zero post-clip fragment gradient has no direction on which to
        # project.  Shadow the hard-reset endpoint and mark the event invalid
        # for mechanism statistics rather than manufacturing a denominator.
        a_raw_tensor = zero.clone()
        a_tensor = zero.clone()
        a_raw = 0.0
        a = 0.0
        fallback_reason = "zero_preconditioned_gradient_energy"

    region = "reset" if a_raw <= 0.0 else "cap" if a_raw >= 1.0 else "in_range"
    totals = {
        key: zero.clone()
        for key in (
            "stock_adaptive_norm2", "ray_adaptive_norm2", "slab_adaptive_norm2",
            "reset_adaptive_norm2", "stock_total_norm2", "ray_total_norm2",
            "slab_total_norm2", "reset_total_norm2", "ray_diff_norm2",
            "slab_diff_norm2", "reset_diff_norm2", "stock_ray_dot",
            "stock_slab_dot", "stock_reset_dot", "stock_current_work",
            "ray_current_work", "slab_current_work", "reset_current_work",
            "ray_raw_change_norm2", "slab_raw_change_norm2", "reset_raw_change_norm2",
            "ray_pre_work", "slab_pre_work", "slab_transverse_preservation_error2",
            "unclipped_transverse_penergy",
        )
    }
    tensor_maps: dict[str, dict[str, torch.Tensor]] | None = (
        {
            "stock_step": {}, "ray_step": {}, "slab_step": {}, "reset_step": {},
            "ray_minus_stock": {}, "slab_minus_stock": {}, "reset_minus_stock": {},
            "ray_raw_exp_avg": {}, "slab_raw_exp_avg": {}, "reset_raw_exp_avg": {},
        }
        if capture_mode is not None
        else None
    )

    for entry in entries:
        bc1 = 1.0 - entry.beta1_power
        ray_m_hat = a_tensor * entry.grad
        slab_m_hat = entry.m_hat + (a_tensor - a_raw_tensor) * entry.grad
        ray_raw = bc1 * ray_m_hat
        slab_raw = bc1 * slab_m_hat
        reset_raw = torch.zeros_like(entry.raw_m)

        bc1_next = 1.0 - entry.beta1 ** (entry.step + 1)
        stock_next = entry.beta1 * entry.raw_m + (1.0 - entry.beta1) * entry.grad
        ray_next = entry.beta1 * ray_raw + (1.0 - entry.beta1) * entry.grad
        slab_next = entry.beta1 * slab_raw + (1.0 - entry.beta1) * entry.grad
        reset_next = (1.0 - entry.beta1) * entry.grad
        stock_adaptive = entry.lr * (stock_next / bc1_next) * entry.preconditioner
        ray_adaptive = entry.lr * (ray_next / bc1_next) * entry.preconditioner
        slab_adaptive = entry.lr * (slab_next / bc1_next) * entry.preconditioner
        reset_adaptive = entry.lr * (reset_next / bc1_next) * entry.preconditioner
        decay = entry.lr * entry.weight_decay * entry.param.detach().to(dtype=accum_dtype)
        stock_total = decay + stock_adaptive
        ray_total = decay + ray_adaptive
        slab_total = decay + slab_adaptive
        reset_total = decay + reset_adaptive
        ray_diff = ray_total - stock_total
        slab_diff = slab_total - stock_total
        reset_diff = reset_total - stock_total

        totals["stock_adaptive_norm2"] += torch.sum(stock_adaptive.square())
        totals["ray_adaptive_norm2"] += torch.sum(ray_adaptive.square())
        totals["slab_adaptive_norm2"] += torch.sum(slab_adaptive.square())
        totals["reset_adaptive_norm2"] += torch.sum(reset_adaptive.square())
        totals["stock_total_norm2"] += torch.sum(stock_total.square())
        totals["ray_total_norm2"] += torch.sum(ray_total.square())
        totals["slab_total_norm2"] += torch.sum(slab_total.square())
        totals["reset_total_norm2"] += torch.sum(reset_total.square())
        totals["ray_diff_norm2"] += torch.sum(ray_diff.square())
        totals["slab_diff_norm2"] += torch.sum(slab_diff.square())
        totals["reset_diff_norm2"] += torch.sum(reset_diff.square())
        totals["stock_ray_dot"] += torch.sum(stock_total * ray_total)
        totals["stock_slab_dot"] += torch.sum(stock_total * slab_total)
        totals["stock_reset_dot"] += torch.sum(stock_total * reset_total)
        totals["stock_current_work"] += torch.sum(entry.grad * stock_total)
        totals["ray_current_work"] += torch.sum(entry.grad * ray_total)
        totals["slab_current_work"] += torch.sum(entry.grad * slab_total)
        totals["reset_current_work"] += torch.sum(entry.grad * reset_total)
        totals["ray_raw_change_norm2"] += torch.sum((ray_raw - entry.raw_m).square())
        totals["slab_raw_change_norm2"] += torch.sum((slab_raw - entry.raw_m).square())
        totals["reset_raw_change_norm2"] += torch.sum(entry.raw_m.square())
        totals["ray_pre_work"] += torch.sum(
            entry.grad * entry.preconditioner * ray_m_hat
        )
        totals["slab_pre_work"] += torch.sum(
            entry.grad * entry.preconditioner * slab_m_hat
        )
        old_transverse = entry.m_hat - a_raw_tensor * entry.grad
        slab_transverse = slab_m_hat - a_tensor * entry.grad
        totals["slab_transverse_preservation_error2"] += torch.sum(
            (slab_transverse - old_transverse).square()
        )
        totals["unclipped_transverse_penergy"] += torch.sum(
            old_transverse * entry.preconditioner * old_transverse
        )

        if tensor_maps is not None:
            state_dtype = (
                optimizer.state[entry.param]["exp_avg"].dtype
                if optimizer.state.get(entry.param)
                else entry.param.dtype
            )
            tensor_maps["ray_minus_stock"][entry.name] = ray_diff.detach().clone()
            tensor_maps["slab_minus_stock"][entry.name] = slab_diff.detach().clone()
            tensor_maps["reset_minus_stock"][entry.name] = reset_diff.detach().clone()
            if capture_mode == "all":
                tensor_maps["stock_step"][entry.name] = stock_total.detach().clone()
                tensor_maps["ray_step"][entry.name] = ray_total.detach().clone()
                tensor_maps["slab_step"][entry.name] = slab_total.detach().clone()
                tensor_maps["reset_step"][entry.name] = reset_total.detach().clone()
                tensor_maps["ray_raw_exp_avg"][entry.name] = (
                    ray_raw.detach().to(dtype=state_dtype).clone()
                )
                tensor_maps["slab_raw_exp_avg"][entry.name] = (
                    slab_raw.detach().to(dtype=state_dtype).clone()
                )
                tensor_maps["reset_raw_exp_avg"][entry.name] = (
                    reset_raw.detach().to(dtype=state_dtype).clone()
                )

    stock_total_l2 = _safe_sqrt(totals["stock_total_norm2"])
    ray_total_l2 = _safe_sqrt(totals["ray_total_norm2"])
    slab_total_l2 = _safe_sqrt(totals["slab_total_norm2"])
    reset_total_l2 = _safe_sqrt(totals["reset_total_norm2"])
    stock_ray_cos = (
        _json_float(totals["stock_ray_dot"]) / (stock_total_l2 * ray_total_l2)
        if stock_total_l2 > 0.0 and ray_total_l2 > 0.0
        else 0.0
    )
    stock_slab_cos = (
        _json_float(totals["stock_slab_dot"]) / (stock_total_l2 * slab_total_l2)
        if stock_total_l2 > 0.0 and slab_total_l2 > 0.0
        else 0.0
    )
    stock_reset_cos = (
        _json_float(totals["stock_reset_dot"]) / (stock_total_l2 * reset_total_l2)
        if stock_total_l2 > 0.0 and reset_total_l2 > 0.0
        else 0.0
    )
    old_penergy = _json_float(old_pnorm2)
    transverse_penergy = _json_float(totals["unclipped_transverse_penergy"])
    event_id = (
        f"l{learner_id}-r{rank}-f{fragment_id}-v{broadcast_version}-"
        f"b{broadcast_local_step}-g{gradient_local_step}"
    )
    record: dict[str, Any] = {
        "schema": SHADOW_SCHEMA,
        "behavior": "shadow_only",
        "policies": ["stock", "ray", "slab", "reset"],
        "accumulation_dtype": str(accum_dtype).removeprefix("torch."),
        "event_id": event_id,
        "learner_id": int(learner_id),
        "rank": int(rank),
        "fragment": int(fragment_id),
        "broadcast_version": int(broadcast_version),
        "broadcast_local_step": int(broadcast_local_step),
        "gradient_local_step": int(gradient_local_step),
        "completed_steps_between_broadcast_and_gradient": int(
            gradient_local_step - broadcast_local_step
        ),
        "upcoming_optimizer_step": int(gradient_local_step + 1),
        "tensor_count": len(entries),
        "numel": int(fragment.numel),
        "optimizer_group_count": len(group_ids),
        "state_step_min": min(steps),
        "state_step_max": max(steps),
        "gradient_l2": _safe_sqrt(grad_norm2),
        "parameter_l2": _safe_sqrt(param_norm2),
        "old_preconditioned_moment_energy": old_penergy,
        "unclipped_transverse_preconditioned_energy": transverse_penergy,
        "unclipped_transverse_energy_fraction": (
            transverse_penergy / old_penergy if old_penergy > 0.0 else 0.0
        ),
        "projection_numerator": n_value,
        "projection_denominator": d_value,
        "a_raw": a_raw,
        "a": a,
        "projection_region": region,
        "fallback": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "ray_preconditioned_work": _json_float(totals["ray_pre_work"]),
        "slab_preconditioned_work": _json_float(totals["slab_pre_work"]),
        "ray_preconditioned_work_target": a * d_value,
        "slab_preconditioned_work_target": min(max(n_value, 0.0), d_value),
        "slab_transverse_preservation_error_l2": _safe_sqrt(
            totals["slab_transverse_preservation_error2"]
        ),
        "stock_adaptive_step_l2": _safe_sqrt(totals["stock_adaptive_norm2"]),
        "ray_adaptive_step_l2": _safe_sqrt(totals["ray_adaptive_norm2"]),
        "slab_adaptive_step_l2": _safe_sqrt(totals["slab_adaptive_norm2"]),
        "reset_adaptive_step_l2": _safe_sqrt(totals["reset_adaptive_norm2"]),
        "stock_total_step_l2": stock_total_l2,
        "ray_total_step_l2": ray_total_l2,
        "slab_total_step_l2": slab_total_l2,
        "reset_total_step_l2": reset_total_l2,
        "ray_minus_stock_step_l2": _safe_sqrt(totals["ray_diff_norm2"]),
        "slab_minus_stock_step_l2": _safe_sqrt(totals["slab_diff_norm2"]),
        "reset_minus_stock_step_l2": _safe_sqrt(totals["reset_diff_norm2"]),
        "stock_ray_step_cosine": max(-1.0, min(1.0, stock_ray_cos)),
        "stock_slab_step_cosine": max(-1.0, min(1.0, stock_slab_cos)),
        "stock_reset_step_cosine": max(-1.0, min(1.0, stock_reset_cos)),
        "stock_current_gradient_work": _json_float(totals["stock_current_work"]),
        "ray_current_gradient_work": _json_float(totals["ray_current_work"]),
        "slab_current_gradient_work": _json_float(totals["slab_current_work"]),
        "reset_current_gradient_work": _json_float(totals["reset_current_work"]),
        "ray_raw_moment_change_l2": _safe_sqrt(totals["ray_raw_change_norm2"]),
        "slab_raw_moment_change_l2": _safe_sqrt(totals["slab_raw_change_norm2"]),
        "reset_raw_moment_change_l2": _safe_sqrt(totals["reset_raw_change_norm2"]),
    }
    if jump_stats:
        for key, value in jump_stats.items():
            if not isinstance(key, str) or not key.startswith("broadcast_"):
                raise ValueError("jump_stats keys must be strings prefixed with 'broadcast_'")
            record[key] = _json_float(value)

    # This also rejects accidental NaN/Infinity before a record reaches disk.
    json.dumps(record, allow_nan=False, sort_keys=True)
    tensors = BCMPShadowTensors(**tensor_maps) if tensor_maps is not None else None
    return BCMPShadowResult(record=record, tensors=tensors)


def score_directions(
    directions: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    *,
    event_id: str,
    candidate: str,
    resolved_local_step: int,
) -> dict[str, Any]:
    """Score a candidate-minus-stock displacement against a future gradient.

    Positive dot/cosine means that subtracting the candidate displacement would
    be more descending under the future gradient than subtracting the stock
    displacement.  Gradients must already be all-reduced and clipped.
    """
    if not directions:
        raise ValueError("cannot resolve an empty direction mapping")
    first = next(iter(directions.values()))
    # Runtime directions are FP32; preserving that dtype avoids introducing
    # slow CUDA FP64 work solely for diagnostics.  FP64 unit problems retain
    # FP64 accumulation, while lower-precision directions promote to FP32.
    dtype = torch.float64 if first.dtype == torch.float64 else torch.float32
    dot = torch.zeros((), dtype=dtype, device=first.device)
    direction_norm2 = dot.clone()
    gradient_norm2 = dot.clone()
    numel = 0
    for name, direction in directions.items():
        if name not in params:
            raise KeyError(f"future-gradient parameter {name!r} is missing")
        grad = params[name].grad
        if grad is None:
            raise ValueError(f"future-gradient parameter {name!r} has no gradient")
        d = direction.detach().to(dtype=dtype)
        g = grad.detach().to(device=d.device, dtype=dtype)
        if d.shape != g.shape:
            raise ValueError(f"future-gradient shape mismatch for {name}")
        dot += torch.sum(g * d)
        direction_norm2 += torch.sum(d.square())
        gradient_norm2 += torch.sum(g.square())
        numel += d.numel()
    direction_l2 = _safe_sqrt(direction_norm2)
    gradient_l2 = _safe_sqrt(gradient_norm2)
    dot_value = _json_float(dot)
    cosine = (
        dot_value / (direction_l2 * gradient_l2)
        if direction_l2 > 0.0 and gradient_l2 > 0.0
        else 0.0
    )
    record = {
        "schema": RESOLUTION_SCHEMA,
        "event_id": str(event_id),
        "candidate": str(candidate),
        "resolved_local_step": int(resolved_local_step),
        "numel": int(numel),
        "future_gradient_dot": dot_value,
        "future_gradient_cosine": max(-1.0, min(1.0, cosine)),
        "direction_l2": direction_l2,
        "future_gradient_l2": gradient_l2,
    }
    json.dumps(record, allow_nan=False, sort_keys=True)
    return record


def append_jsonl(path: str | os.PathLike[str], record: dict[str, Any]) -> None:
    """Append one finite, canonical JSON record.

    Each learner/rank should use its own file.  This helper intentionally does
    not pretend to provide cross-process locking.
    """
    encoded = json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":"))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


class BCMPShadowTracker:
    """Runtime lifecycle for sampled broadcast shadows.

    The tracker exists on the syncer-facing rank only.  ``note_broadcast`` is
    called immediately before the learner applies/blends a received fragment.
    ``before_optimizer_step`` is called after all-reduce/clipping and before
    factual AdamW.  Every sampled counterfactual is resolved against the next
    clipped gradient, then discarded.  None of these operations mutates live
    training state.

    The log starts fresh for each learner process.  Current Yeto learner
    checkpoints do not include optimizer state, so pretending to resume a
    pending shadow across process replacement would be less reproducible than
    explicitly logging it as dropped.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        every: int,
        learner_id: int,
        rank: int,
    ) -> None:
        if every < 1:
            raise ValueError(f"BC-MP shadow cadence must be >= 1, got {every}")
        self.path = Path(path)
        self.every = int(every)
        self.learner_id = int(learner_id)
        self.rank = int(rank)
        self.broadcasts_seen = 0
        self.broadcasts_sampled = 0
        self.step_hooks = 0
        self.shadow_events_recorded = 0
        self.resolutions_recorded = 0
        self.drops_recorded = 0
        self.note_broadcast_wall_s = 0.0
        self.before_optimizer_step_wall_s = 0.0
        self.started_at = time.perf_counter()
        self.pending_broadcasts: dict[int, dict[str, Any]] = {}
        self.pending_resolutions: list[
            tuple[str, str, int, dict[str, torch.Tensor]]
        ] = []
        self.closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        append_jsonl(
            self.path,
            {
                "schema": RUN_SCHEMA,
                "learner_id": self.learner_id,
                "rank": self.rank,
                "every": self.every,
            },
        )

    def _drop(self, *, reason: str, event: dict[str, Any], local_step: int) -> None:
        append_jsonl(
            self.path,
            {
                "schema": DROP_SCHEMA,
                "learner_id": self.learner_id,
                "rank": self.rank,
                "fragment": int(event["fragment"]),
                "broadcast_version": int(event["broadcast_version"]),
                "broadcast_local_step": int(event["broadcast_local_step"]),
                "drop_local_step": int(local_step),
                "reason": reason,
            },
        )
        self.drops_recorded += 1

    def _drop_resolution(
        self,
        *,
        event_id: str,
        candidate: str,
        fragment_id: int,
        reason: str,
        local_step: int,
    ) -> None:
        append_jsonl(
            self.path,
            {
                "schema": DROP_SCHEMA,
                "event_id": event_id,
                "candidate": candidate,
                "learner_id": self.learner_id,
                "rank": self.rank,
                "fragment": int(fragment_id),
                "drop_local_step": int(local_step),
                "reason": reason,
            },
        )
        self.drops_recorded += 1

    def note_broadcast(
        self,
        *,
        fragment_id: int,
        broadcast_version: int,
        local_step: int,
        fragment: Fragment,
        params: dict[str, torch.Tensor],
        global_flat: torch.Tensor,
        merge_alpha: float,
    ) -> bool:
        """Register an about-to-be-applied broadcast; return whether sampled."""
        started = time.perf_counter()
        if self.closed:
            raise RuntimeError("cannot note a broadcast on a closed BC-MP shadow tracker")
        self.broadcasts_seen += 1
        # A second broadcast before the next-gradient resolution changes the
        # evaluation point, so that gradient is not an uncontaminated one-step
        # lookahead for the older candidate direction.
        keep_resolutions = []
        for event_id, candidate, resolution_fid, directions in self.pending_resolutions:
            if resolution_fid == int(fragment_id):
                self._drop_resolution(
                    event_id=event_id,
                    candidate=candidate,
                    fragment_id=resolution_fid,
                    reason="rebroadcast_before_future_gradient",
                    local_step=local_step,
                )
            else:
                keep_resolutions.append(
                    (event_id, candidate, resolution_fid, directions)
                )
        self.pending_resolutions = keep_resolutions
        # Even an unsampled newer apply invalidates a sampled event for this
        # fragment: the next gradient would be relative to the newer parameter
        # value, not the sampled broadcast's value.
        old = self.pending_broadcasts.pop(int(fragment_id), None)
        if old is not None:
            self._drop(
                reason="superseded_before_fresh_gradient",
                event=old,
                local_step=local_step,
            )
        if (self.broadcasts_seen - 1) % self.every != 0:
            self.note_broadcast_wall_s += time.perf_counter() - started
            return False
        self.broadcasts_sampled += 1
        self.pending_broadcasts[int(fragment_id)] = {
            "fragment": int(fragment_id),
            "broadcast_version": int(broadcast_version),
            "broadcast_local_step": int(local_step),
            "jump_stats": broadcast_jump_stats(
                fragment, params, global_flat, merge_alpha
            ),
        }
        self.note_broadcast_wall_s += time.perf_counter() - started
        return True

    def before_optimizer_step(
        self,
        *,
        layout: Any,
        params: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        local_step: int,
    ) -> int:
        """Resolve old events and record new first-post-broadcast shadows.

        Returns the number of newly recorded shadow events.
        """
        started = time.perf_counter()
        if self.closed:
            raise RuntimeError("cannot step a closed BC-MP shadow tracker")
        self.step_hooks += 1
        # Resolve first: these directions came from the previous optimizer
        # boundary, so the current post-clip gradient is genuinely one step in
        # the future.  Resolution must precede construction of new events.
        for event_id, candidate, _fragment_id, directions in self.pending_resolutions:
            resolution = score_directions(
                directions,
                params,
                event_id=event_id,
                candidate=candidate,
                resolved_local_step=local_step,
            )
            append_jsonl(self.path, resolution)
            self.resolutions_recorded += 1
        self.pending_resolutions.clear()

        recorded = 0
        pending = sorted(
            self.pending_broadcasts.values(),
            key=lambda event: (event["broadcast_local_step"], event["fragment"]),
        )
        self.pending_broadcasts.clear()
        for event in pending:
            fid = int(event["fragment"])
            result = compute_bcmp_shadow(
                layout.fragments[fid],
                params,
                optimizer,
                fragment_id=fid,
                broadcast_version=int(event["broadcast_version"]),
                broadcast_local_step=int(event["broadcast_local_step"]),
                gradient_local_step=int(local_step),
                learner_id=self.learner_id,
                rank=self.rank,
                jump_stats=event["jump_stats"],
                capture_tensors="directions",
                accum_dtype=torch.float32,
            )
            append_jsonl(self.path, result.record)
            if result.tensors is None:  # defensive: directions requested above
                raise RuntimeError("BC-MP shadow failed to retain resolution directions")
            self.pending_resolutions.extend(
                [
                    (result.record["event_id"], "ray", fid, result.tensors.ray_minus_stock),
                    (result.record["event_id"], "slab", fid, result.tensors.slab_minus_stock),
                    (result.record["event_id"], "reset", fid, result.tensors.reset_minus_stock),
                ]
            )
            recorded += 1
            self.shadow_events_recorded += 1
        self.before_optimizer_step_wall_s += time.perf_counter() - started
        return recorded

    def close(self, *, local_step: int) -> None:
        """Log unresolved tail events and release retained tensors."""
        if self.closed:
            return
        for event in self.pending_broadcasts.values():
            self._drop(
                reason="shutdown_before_fresh_gradient",
                event=event,
                local_step=local_step,
            )
        self.pending_broadcasts.clear()
        for event_id, candidate, fragment_id, _directions in self.pending_resolutions:
            self._drop_resolution(
                event_id=event_id,
                candidate=candidate,
                fragment_id=fragment_id,
                reason="shutdown_before_future_gradient",
                local_step=local_step,
            )
        self.pending_resolutions.clear()
        active_wall_s = time.perf_counter() - self.started_at
        shadow_wall_s = (
            self.note_broadcast_wall_s + self.before_optimizer_step_wall_s
        )
        ratio = shadow_wall_s / active_wall_s if active_wall_s > 0.0 else 0.0
        summary = {
            "schema": RUN_SUMMARY_SCHEMA,
            "learner_id": self.learner_id,
            "rank": self.rank,
            "every": self.every,
            "broadcasts_seen": self.broadcasts_seen,
            "broadcasts_sampled": self.broadcasts_sampled,
            "step_hooks": self.step_hooks,
            "shadow_events_recorded": self.shadow_events_recorded,
            "resolutions_recorded": self.resolutions_recorded,
            "drops_recorded": self.drops_recorded,
            "note_broadcast_wall_s": self.note_broadcast_wall_s,
            "before_optimizer_step_wall_s": self.before_optimizer_step_wall_s,
            "shadow_wall_s": shadow_wall_s,
            "active_wall_s": active_wall_s,
            "shadow_wall_fraction": ratio,
            "cuda_synchronized_for_timing": False,
            "factual_optimizer_state_mutated": False,
            "async_final_loss_parity_caveat": (
                "host overhead can perturb non-barrier broadcast arrival/order timing"
            ),
        }
        json.dumps(summary, allow_nan=False, sort_keys=True)
        append_jsonl(self.path, summary)
        self.closed = True
