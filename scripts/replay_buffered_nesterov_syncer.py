#!/usr/bin/env python3
"""Replay buffered robust policies through the production outer optimizer.

Unlike the earlier buffered screen, this replay uses the syncer's actual
current-group baseline: per-tensor HeLoCo correction, Avg/RDA aggregation,
and the configured Nesterov outer step. Buffered policies replace the merge
gradient but use the same Nesterov state and never inspect probe utility when
constructing an update.

The captured trajectory still comes from the token-weighted baseline, so each
record is a one-step counterfactual rather than a full counterfactual rollout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


buffered = _load_script("replay_buffered_robust_syncer")
soft = buffered.soft
syncer_eval = buffered.syncer_eval

from yeto.export import parse_checkpoint  # noqa: E402
from yeto.fragments import MERGE_AVG, build_layout  # noqa: E402
from yeto.learner import load_model_and_tokenizer, trainable_params  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402
from yeto.tensor_io import apply_fragment  # noqa: E402


POLICIES = (
    "token_weighted",
    "current_token_scale50",
    "current_token_scale75",
    "current_token_scale90",
    "current_outer_lr25",
    "current_outer_lr35",
    "current_outer_lr40",
    "current_outer_lr50",
    "current_outer_lr60",
    "current_outer_lr75",
    "current_outer_lr90",
    "current_avg_heloco",
    "current_consensus_rda_sqrt",
    "current_consensus_rda_linear",
    "current_consensus_rda_affine50",
    "current_consensus_rda_floor50",
    "current_consensus_outer_lr_p1",
    "current_consensus_outer_lr_p15",
    "current_consensus_outer_lr_p2",
    "current_avg_norm_outer_lr",
    "current_median_norm_outer_lr",
    "current_median_norm_outer_lr_p15",
    "current_coord_midpoint_raw",
    "current_coord_midpoint_heloco",
    "current_coord_midpoint_normmatch",
    "current_rda_median_norm",
    "current_coord_midpoint_blend25",
    "current_coord_midpoint_blend50",
    "current_geomedian_heloco",
    "buffer_coord_midpoint_raw",
    "buffer_coord_midpoint_heloco",
    "buffer_transport_age4",
    "buffer_transport_age8",
    "buffer_transport_huber_age4",
    "buffer_transport_huber_age8",
    "buffer_transport_wcoordmed25_age4",
    "buffer_transport_wcoordmed50_age4",
    "buffer_group_ema10",
    "buffer_group_ema25",
    "buffer_group_ema40",
    "buffer_group_transport25",
    "buffer_group_guard25",
    "buffer_group_clip25",
    "buffer_group_coord_midpoint",
)


@dataclass
class Buffered:
    row: dict
    update: torch.Tensor
    weight: float


def _mean(values: list[float]) -> float:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(vals) / len(vals) if vals else float("nan")


def _quantile(values: list[float], p: float) -> float:
    vals = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not vals:
        return float("nan")
    return vals[max(0, min(len(vals) - 1, round(p * (len(vals) - 1))))]


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _infer_seed(*paths: Path | None) -> int | None:
    for path in paths:
        if path is None:
            continue
        match = re.search(r"seed(\d+)", str(path))
        if match:
            return int(match.group(1))
    return None


def _weighted(updates: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    total = sum(max(float(weight), 0.0) for weight in weights if math.isfinite(float(weight)))
    if total <= 0.0:
        return torch.zeros_like(updates[0])
    out = torch.zeros_like(updates[0])
    for update, weight in zip(updates, weights):
        if weight > 0.0 and math.isfinite(weight):
            out.add_(update, alpha=float(weight))
    return out.div_(total)


def _weighted_median(values: torch.Tensor, weights: list[float]) -> torch.Tensor:
    """Coordinate-wise weighted median for [candidate, coordinate] values."""
    order = values.argsort(dim=0)
    sorted_values = values.gather(0, order)
    weight_tensor = torch.tensor(weights, dtype=values.dtype, device=values.device)
    expanded = weight_tensor[:, None].expand_as(values)
    sorted_weights = expanded.gather(0, order)
    cutoff = max(sum(max(float(weight), 0.0) for weight in weights), 1e-12) * 0.5
    cumulative = sorted_weights.cumsum(dim=0)
    positions = (cumulative >= cutoff).to(torch.int64).argmax(dim=0)
    selected = sorted_values.gather(0, positions.unsqueeze(0)).squeeze(0)
    at_cutoff = cumulative.gather(0, positions.unsqueeze(0)).squeeze(0)
    tied = torch.isclose(
        at_cutoff,
        torch.as_tensor(cutoff, dtype=values.dtype, device=values.device),
        rtol=1e-6,
        atol=1e-12,
    ) & (positions < values.shape[0] - 1)
    next_positions = (positions + 1).clamp(max=values.shape[0] - 1)
    upper = sorted_values.gather(0, next_positions.unsqueeze(0)).squeeze(0)
    return torch.where(tied, 0.5 * (selected + upper), selected)


def _coordinate_midpoint_median(values: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(values).sort(dim=0).values
    n = stacked.shape[0]
    if n % 2:
        return stacked[n // 2]
    return 0.5 * (stacked[n // 2 - 1] + stacked[n // 2])


def _weighted_scalar_median(values: list[float], weights: list[float]) -> float:
    pairs = sorted(zip(values, weights), key=lambda pair: pair[0])
    total = sum(max(weight, 0.0) for _, weight in pairs)
    if total <= 0.0:
        return _quantile(values, 0.5)
    running = 0.0
    for value, weight in pairs:
        running += max(weight, 0.0)
        if running >= 0.5 * total:
            return float(value)
    return float(pairs[-1][0])


def _cap_history(weights: list[float], relative_ages: list[float], cap: float) -> list[float]:
    out = [max(float(weight), 0.0) for weight in weights]
    fresh = sum(weight for weight, age in zip(out, relative_ages) if age <= 0.0)
    history = sum(weight for weight, age in zip(out, relative_ages) if age > 0.0)
    if fresh <= 0.0 or history <= 0.0 or cap >= 1.0:
        return out
    allowed = cap / max(1.0 - cap, 1e-12) * fresh
    if history > allowed:
        scale = allowed / history
        out = [weight if age <= 0.0 else weight * scale for weight, age in zip(out, relative_ages)]
    return out


def _weight_stats(weights: list[float], relative_ages: list[float]) -> dict:
    total = sum(max(weight, 0.0) for weight in weights)
    sq = sum(max(weight, 0.0) ** 2 for weight in weights)
    fresh = sum(max(weight, 0.0) for weight, age in zip(weights, relative_ages) if age <= 0.0)
    ess = 0.0 if sq <= 0.0 else total * total / sq
    return {
        "effective_sample_size": ess,
        "normalized_effective_sample_size": ess / max(len(weights), 1),
        "fresh_effective_share": 0.0 if total <= 0.0 else fresh / total,
        "history_effective_share": 0.0 if total <= 0.0 else 1.0 - fresh / total,
    }


def _heloco_per_tensor(update: torch.Tensor, momentum: torch.Tensor, tensor_numels: list[int]) -> torch.Tensor:
    out = update.clone()
    offset = 0
    for numel in tensor_numels:
        end = offset + int(numel)
        out[offset:end] = soft._heloco_update(out[offset:end], momentum[offset:end])
        offset = end
    return out


def _tensor_numels(frag) -> list[int]:
    return [int(numel) for _, numel in frag.tensors]


def _rda(updates: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    total = sum(max(weight, 0.0) for weight in weights)
    if total <= 0.0:
        return torch.zeros_like(updates[0])
    norms = [float(update.norm().item()) for update in updates]
    radial = sum(max(weight, 0.0) * norm for weight, norm in zip(weights, norms)) / total
    direction = torch.zeros_like(updates[0])
    for update, weight, norm in zip(updates, weights, norms):
        if weight > 0.0 and norm > 0.0:
            direction.add_(update, alpha=weight / total / norm)
    norm = float(direction.norm().item())
    if norm < 1e-12:
        return _weighted(updates, weights)
    return direction.mul_(radial / norm)


def _production_merge_update(candidates: list[soft.Candidate], momentum: torch.Tensor, frag) -> torch.Tensor:
    updates = [
        _heloco_per_tensor(candidate.update, momentum, _tensor_numels(frag))
        for candidate in candidates
    ]
    weights = [candidate.weight for candidate in candidates]
    out = torch.empty_like(updates[0])
    offset = 0
    for numel in _tensor_numels(frag):
        end = offset + int(numel)
        slices = [update[offset:end] for update in updates]
        if frag.merge_mode == MERGE_AVG:
            out[offset:end] = _weighted(slices, weights)
        else:
            out[offset:end] = _rda(slices, weights)
        offset = end
    return out


def _production_avg_update(candidates: list[soft.Candidate], momentum: torch.Tensor, frag) -> torch.Tensor:
    updates = [
        _heloco_per_tensor(candidate.update, momentum, _tensor_numels(frag))
        for candidate in candidates
    ]
    return _weighted(updates, [candidate.weight for candidate in candidates])


def _consensus_rda_update(
    candidates: list[soft.Candidate],
    momentum: torch.Tensor,
    frag,
    mode: str,
) -> tuple[torch.Tensor, float]:
    corrected = [
        _heloco_per_tensor(candidate.update, momentum, _tensor_numels(frag))
        for candidate in candidates
    ]
    weights = [candidate.weight for candidate in candidates]
    if frag.merge_mode == MERGE_AVG:
        return _weighted(corrected, weights), 1.0
    out = torch.empty_like(corrected[0])
    scales = []
    offset = 0
    for numel in _tensor_numels(frag):
        end = offset + numel
        slices = [update[offset:end] for update in corrected]
        total = sum(max(weight, 0.0) for weight in weights)
        norms = [float(update.norm().item()) for update in slices]
        radial = sum(max(weight, 0.0) * norm for weight, norm in zip(weights, norms)) / max(total, 1e-12)
        direction = torch.zeros_like(slices[0])
        for update, weight, norm in zip(slices, weights, norms):
            if weight > 0.0 and norm > 1e-12:
                direction.add_(update, alpha=weight / max(total, 1e-12) / norm)
        consensus = min(1.0, max(0.0, float(direction.norm().item())))
        if consensus < 1e-12:
            out[offset:end] = _weighted(slices, weights)
            scales.append(0.0)
            offset = end
            continue
        if mode == "sqrt":
            scale = math.sqrt(consensus)
        elif mode == "linear":
            scale = consensus
        elif mode == "affine50":
            scale = 0.5 + 0.5 * consensus
        elif mode == "floor50":
            scale = max(0.5, consensus)
        else:
            raise ValueError(mode)
        out[offset:end] = direction * (radial * scale / consensus)
        scales.append(scale)
        offset = end
    return out, _mean(scales)


def _direction_consensus(
    candidates: list[soft.Candidate],
    momentum: torch.Tensor,
    frag,
) -> float:
    corrected = [
        _heloco_per_tensor(candidate.update, momentum, _tensor_numels(frag))
        for candidate in candidates
    ]
    weights = [candidate.weight for candidate in candidates]
    total_weight = sum(max(weight, 0.0) for weight in weights)
    weighted_consensus = 0.0
    total_numel = 0
    offset = 0
    for numel in _tensor_numels(frag):
        end = offset + numel
        direction = torch.zeros_like(corrected[0][offset:end])
        for update, weight in zip(corrected, weights):
            update_slice = update[offset:end]
            norm = float(update_slice.norm().item())
            if weight > 0.0 and norm > 1e-12:
                direction.add_(update_slice, alpha=weight / max(total_weight, 1e-12) / norm)
        consensus = min(1.0, max(0.0, float(direction.norm().item())))
        weighted_consensus += consensus * numel
        total_numel += numel
        offset = end
    return weighted_consensus / max(total_numel, 1)


def _match_tensor_norms(source: torch.Tensor, target: torch.Tensor, frag) -> torch.Tensor:
    out = torch.empty_like(source)
    offset = 0
    for numel in _tensor_numels(frag):
        end = offset + numel
        source_slice = source[offset:end]
        source_norm = float(source_slice.norm().item())
        target_norm = float(target[offset:end].norm().item())
        if source_norm < 1e-12:
            out[offset:end] = source_slice
        else:
            out[offset:end] = source_slice * (target_norm / source_norm)
        offset = end
    return out


def _geomedian_per_tensor(
    updates: list[torch.Tensor],
    weights: list[float],
    frag,
) -> torch.Tensor:
    out = torch.empty_like(updates[0])
    offset = 0
    for numel in _tensor_numels(frag):
        end = offset + numel
        out[offset:end] = buffered._geomedian(
            [update[offset:end] for update in updates], weights
        )
        offset = end
    return out


def _nesterov_trial(
    current: torch.Tensor,
    momentum: torch.Tensor,
    merged_update: torch.Tensor,
    outer_lr: float,
    outer_momentum: float,
) -> torch.Tensor:
    # Rust uses delta = current - learner = -merged_update as the gradient.
    delta = -merged_update
    next_momentum = momentum.mul(outer_momentum).add(delta)
    return current - outer_lr * (delta + outer_momentum * next_momentum)


def _transport_slice(
    updates: list[torch.Tensor],
    base_weights: list[float],
    relative_ages: list[float],
    momentum: torch.Tensor,
    *,
    tau: float,
    history_cap: float,
    mode: str,
    median_blend: float = 0.25,
) -> tuple[torch.Tensor, dict]:
    fresh_indices = [idx for idx, age in enumerate(relative_ages) if age <= 0.0]
    fresh_updates = [updates[idx] for idx in fresh_indices]
    fresh_weights = [base_weights[idx] for idx in fresh_indices]
    fresh_mean = _weighted(fresh_updates, fresh_weights)

    pnorm = float(momentum.norm().item())
    direction = None if pnorm < 1e-12 else -momentum / pnorm
    fresh_projection = 0.0 if direction is None else float(torch.dot(fresh_mean, direction).item())
    transported = []
    transport_norms = []
    for update, age in zip(updates, relative_ages):
        moved = update
        if direction is not None and age > 0.0:
            rho = 1.0 - math.exp(-age / tau)
            projection = float(torch.dot(update, direction).item())
            moved = update + rho * (fresh_projection - projection) * direction
        transported.append(moved)
        transport_norms.append(float((moved - update).norm().item()))

    weights = [weight * math.exp(-age / tau) for weight, age in zip(base_weights, relative_ages)]
    weights = _cap_history(weights, relative_ages, history_cap)
    effective = list(weights)
    if mode == "mean":
        merged = _weighted(transported, effective)
    elif mode == "huber":
        total = sum(effective)
        weighted_sum = torch.zeros_like(transported[0])
        for update, weight in zip(transported, effective):
            weighted_sum.add_(update, alpha=weight)
        distances = []
        for update, weight in zip(transported, effective):
            denom = total - weight
            center = weighted_sum.sub(update, alpha=weight).div(max(denom, 1e-12))
            distances.append(float((update - center).norm().item()))
        radius = max(_weighted_scalar_median(distances, effective), 1e-12)
        huber = [max(0.5, min(1.0, 1.5 * radius / max(distance, 1e-12))) for distance in distances]
        effective = _cap_history(
            [weight * factor for weight, factor in zip(effective, huber)],
            relative_ages,
            history_cap,
        )
        merged = _weighted(transported, effective)
    elif mode == "weighted_coord_median":
        median = _weighted_median(torch.stack(transported), effective)
        merged = fresh_mean.mul(1.0 - median_blend).add(median, alpha=median_blend)
    else:
        raise ValueError(mode)

    guard_active = False
    if direction is not None:
        projection = float(torch.dot(merged, direction).item())
        if projection < fresh_projection:
            merged = merged + (fresh_projection - projection) * direction
            guard_active = True
    stats = _weight_stats(effective, relative_ages)
    stats.update(
        {
            "mean_transport_norm": _mean(transport_norms),
            "guard_active": guard_active,
        }
    )
    return merged, stats


def _aggregate(policy: str, candidates: list[soft.Candidate], momentum: torch.Tensor, frag, args) -> tuple[torch.Tensor, dict]:
    updates = [candidate.update for candidate in candidates]
    weights = [candidate.weight for candidate in candidates]
    ages = [candidate.age for candidate in candidates]
    min_age = min(ages)
    relative_ages = [age - min_age for age in ages]
    info = {
        "selected_count": len(candidates),
        "selected_mass": 1.0,
        "mean_age": _mean(ages),
        "p95_age": _quantile(ages, 0.95),
    }
    production = None
    if policy.startswith("current_"):
        production = _production_merge_update(candidates, momentum, frag)
    match = re.fullmatch(r"current_token_scale(\d+)", policy)
    if match:
        info.update(_weight_stats(weights, relative_ages))
        return production * (float(match.group(1)) / 100.0), info
    match = re.fullmatch(r"current_outer_lr(\d+)", policy)
    if match:
        info.update(_weight_stats(weights, relative_ages))
        info["outer_lr_multiplier"] = float(match.group(1)) / 100.0
        return production, info
    if policy == "current_avg_heloco":
        info.update(_weight_stats(weights, relative_ages))
        return _production_avg_update(candidates, momentum, frag), info
    if policy == "current_avg_norm_outer_lr":
        info.update(_weight_stats(weights, relative_ages))
        average = _production_avg_update(candidates, momentum, frag)
        ratio = float(average.norm().item()) / max(float(production.norm().item()), 1e-12)
        info["outer_lr_multiplier"] = max(0.05, min(1.0, ratio))
        return production, info
    match = re.fullmatch(r"current_consensus_rda_(sqrt|linear|affine50|floor50)", policy)
    if match:
        info.update(_weight_stats(weights, relative_ages))
        update, consensus_scale = _consensus_rda_update(
            candidates, momentum, frag, match.group(1)
        )
        info["consensus_scale"] = consensus_scale
        return update, info
    match = re.fullmatch(r"current_consensus_outer_lr_p(1|15|2)", policy)
    if match:
        info.update(_weight_stats(weights, relative_ages))
        consensus = _direction_consensus(candidates, momentum, frag)
        power = {"1": 1.0, "15": 1.5, "2": 2.0}[match.group(1)]
        info["consensus_scale"] = consensus
        info["outer_lr_multiplier"] = max(0.05, consensus**power)
        return production, info
    if policy in {"current_coord_midpoint_raw", "buffer_coord_midpoint_raw"}:
        info.update(_weight_stats(weights, relative_ages))
        return _coordinate_midpoint_median(updates), info
    if policy in {
        "current_coord_midpoint_heloco",
        "current_coord_midpoint_normmatch",
        "current_rda_median_norm",
        "current_coord_midpoint_blend25",
        "current_coord_midpoint_blend50",
        "buffer_coord_midpoint_heloco",
    }:
        corrected = [_heloco_per_tensor(update, momentum, _tensor_numels(frag)) for update in updates]
        median = _coordinate_midpoint_median(corrected)
        info.update(_weight_stats(weights, relative_ages))
        if policy == "current_coord_midpoint_normmatch":
            return _match_tensor_norms(median, production, frag), info
        if policy == "current_rda_median_norm":
            return _match_tensor_norms(production, median, frag), info
        if policy.startswith("current_coord_midpoint_blend"):
            blend = float(policy.removeprefix("current_coord_midpoint_blend")) / 100.0
            return production.mul(1.0 - blend).add(median, alpha=blend), info
        return median, info
    if policy in {"current_median_norm_outer_lr", "current_median_norm_outer_lr_p15"}:
        corrected = [_heloco_per_tensor(update, momentum, _tensor_numels(frag)) for update in updates]
        median = _coordinate_midpoint_median(corrected)
        info.update(_weight_stats(weights, relative_ages))
        ratio = float(median.norm().item()) / max(float(production.norm().item()), 1e-12)
        if policy.endswith("p15"):
            ratio = ratio**1.5
        info["outer_lr_multiplier"] = max(0.05, min(1.0, ratio))
        return production, info
    if policy == "current_geomedian_heloco":
        corrected = [_heloco_per_tensor(update, momentum, _tensor_numels(frag)) for update in updates]
        info.update(_weight_stats(weights, relative_ages))
        return _geomedian_per_tensor(corrected, weights, frag), info

    match = re.fullmatch(r"buffer_transport_(age|huber_age)(\d+)", policy)
    if match:
        mode = "mean" if match.group(1) == "age" else "huber"
        tau = float(match.group(2))
        blend = 0.0
    else:
        match = re.fullmatch(r"buffer_transport_wcoordmed(\d+)_age(\d+)", policy)
        if not match:
            raise ValueError(policy)
        mode = "weighted_coord_median"
        blend = float(match.group(1)) / 100.0
        tau = float(match.group(2))

    out = torch.empty_like(updates[0])
    slice_stats = []
    offset = 0
    for numel in _tensor_numels(frag):
        end = offset + int(numel)
        merged, stats = _transport_slice(
            [update[offset:end] for update in updates],
            weights,
            relative_ages,
            momentum[offset:end],
            tau=tau,
            history_cap=args.history_cap,
            mode=mode,
            median_blend=blend,
        )
        out[offset:end] = merged
        stats["numel"] = int(numel)
        slice_stats.append(stats)
        offset = end
    total_numel = sum(stats["numel"] for stats in slice_stats)
    for key in (
        "effective_sample_size",
        "normalized_effective_sample_size",
        "fresh_effective_share",
        "history_effective_share",
        "mean_transport_norm",
    ):
        info[key] = sum(stats[key] * stats["numel"] for stats in slice_stats) / total_numel
    info["guard_active_fraction"] = _mean([1.0 if stats["guard_active"] else 0.0 for stats in slice_stats])
    return out, info


def _group_policy(
    policy: str,
    rounds: list[list[soft.Candidate]],
    momentum: torch.Tensor,
    frag,
) -> tuple[torch.Tensor, dict]:
    group_updates = [_production_merge_update(group, momentum, frag) for group in rounds]
    current = group_updates[-1]
    history = group_updates[:-1]
    ages = [_mean([candidate.age for candidate in group]) for group in rounds]
    if not history:
        return current, {
            "selected_count": len(rounds[-1]),
            "selected_mass": 1.0,
            "fresh_effective_share": 1.0,
            "history_effective_share": 0.0,
            "normalized_effective_sample_size": 1.0,
            "mean_age": ages[-1],
            "p95_age": ages[-1],
            "guard_active_fraction": 0.0,
        }

    if policy == "buffer_group_coord_midpoint":
        merged = _coordinate_midpoint_median(group_updates)
        fresh_share = 1.0 / len(group_updates)
        guard_fraction = 0.0
    else:
        match = re.fullmatch(r"buffer_group_ema(\d+)", policy)
        alpha = float(match.group(1)) / 100.0 if match else 0.25
        history_weights = [math.exp(-(age - ages[-1]) / 4.0) for age in ages[:-1]]
        history_mean = _weighted(history, history_weights)
        guard_fraction = 0.0
        if policy == "buffer_group_clip25":
            clipped = torch.empty_like(history_mean)
            offset = 0
            for numel in _tensor_numels(frag):
                end = offset + numel
                old_slice = history_mean[offset:end]
                current_slice = current[offset:end]
                scale = min(
                    1.0,
                    1.5 * float(current_slice.norm().item()) / max(float(old_slice.norm().item()), 1e-12),
                )
                clipped[offset:end] = old_slice * scale
                offset = end
            history_mean = clipped
        if policy == "buffer_group_transport25":
            transported = torch.empty_like(history_mean)
            offset = 0
            relative_age = max(0.0, _mean(ages[:-1]) - ages[-1])
            rho = 1.0 - math.exp(-relative_age / 4.0)
            for numel in _tensor_numels(frag):
                end = offset + numel
                old_slice = history_mean[offset:end]
                current_slice = current[offset:end]
                momentum_slice = momentum[offset:end]
                norm = float(momentum_slice.norm().item())
                if norm < 1e-12:
                    transported[offset:end] = old_slice
                else:
                    direction = -momentum_slice / norm
                    old_projection = float(torch.dot(old_slice, direction).item())
                    current_projection = float(torch.dot(current_slice, direction).item())
                    transported[offset:end] = (
                        old_slice + rho * (current_projection - old_projection) * direction
                    )
                offset = end
            history_mean = transported
        merged = current.mul(1.0 - alpha).add(history_mean, alpha=alpha)
        if policy == "buffer_group_guard25":
            guarded = merged.clone()
            active = 0
            offset = 0
            tensor_numels = _tensor_numels(frag)
            for numel in tensor_numels:
                end = offset + numel
                momentum_slice = momentum[offset:end]
                norm = float(momentum_slice.norm().item())
                if norm >= 1e-12:
                    direction = -momentum_slice / norm
                    current_projection = float(torch.dot(current[offset:end], direction).item())
                    merged_projection = float(torch.dot(guarded[offset:end], direction).item())
                    if merged_projection < current_projection:
                        guarded[offset:end].add_(
                            direction, alpha=current_projection - merged_projection
                        )
                        active += 1
                offset = end
            merged = guarded
            guard_fraction = active / max(len(tensor_numels), 1)
        fresh_share = 1.0 - alpha

    effective = [fresh_share]
    if history:
        effective.extend([(1.0 - fresh_share) / len(history)] * len(history))
    ess = 1.0 / sum(weight * weight for weight in effective)
    return merged, {
        "selected_count": sum(len(group) for group in rounds),
        "selected_mass": 1.0,
        "fresh_effective_share": fresh_share,
        "history_effective_share": 1.0 - fresh_share,
        "effective_sample_size": ess,
        "normalized_effective_sample_size": ess / len(effective),
        "mean_age": _mean(ages),
        "p95_age": _quantile(ages, 0.95),
        "guard_active_fraction": guard_fraction,
    }


def _utility_se(base_by_batch: list[float], trial_by_batch: list[float]) -> float | None:
    utilities = [base - trial for base, trial in zip(base_by_batch, trial_by_batch)]
    if len(utilities) < 2:
        return None
    avg = sum(utilities) / len(utilities)
    variance = sum((utility - avg) ** 2 for utility in utilities) / (len(utilities) - 1)
    return math.sqrt(variance / len(utilities))


def _eval(model, batches, compute_loss, frag, params, current, trial, base_loss, base_by_batch, device):
    apply_fragment(frag, trial.to(device), params)
    trial_loss, trial_by_batch = syncer_eval._losses(model, batches, compute_loss)
    apply_fragment(frag, current.to(device), params)
    utility = base_loss - trial_loss
    return utility, _utility_se(base_by_batch, trial_by_batch)


def _next_state_paths(groups: list[list[dict]], root: Path) -> dict[tuple[int, int], Path]:
    out = {}
    for group, next_group in zip(groups, groups[1:]):
        fid = int(group[0]["fragment"])
        out[(int(group[0]["step"]), fid)] = buffered._resolve(
            root, next_group[0]["state_checkpoint"]
        )
    return out


def _buffered_candidate(
    item: Buffered,
    current: torch.Tensor,
    momentum: torch.Tensor,
    current_version: int,
) -> soft.Candidate:
    return soft.Candidate(
        row=item.row,
        tensor=current + item.update,
        update=item.update,
        weight=item.weight,
        age=max(0.0, float(current_version) - float(item.row.get("base_version", current_version))),
        norm=float(item.update.norm().item()),
        align=soft._cosine(item.update, -momentum),
    )


def replay(args) -> list[dict]:
    root = args.capture_dir or args.index.parent
    index = args.index or root / "index.jsonl"
    all_groups = buffered._group_rows(buffered._read_jsonl(index), 1)
    incomplete = []
    for group in all_groups:
        learners = [int(row["learner_id"]) for row in group]
        if len(learners) != len(set(learners)):
            raise SystemExit(
                f"duplicate learner in step={group[0]['step']} fragment={group[0]['fragment']}"
            )
        if args.expected_candidates and len(group) != args.expected_candidates:
            incomplete.append(group)
    if incomplete and not args.drop_incomplete_groups:
        first = incomplete[0]
        raise SystemExit(
            f"incomplete group step={first[0]['step']} fragment={first[0]['fragment']}: "
            f"got {len(first)}, expected {args.expected_candidates}; "
            "pass --drop-incomplete-groups only when these are known terminal partial rounds"
        )
    groups = [
        group
        for group in all_groups
        if (
            len(group) == args.expected_candidates
            if args.expected_candidates
            else len(group) >= args.min_candidates
        )
    ]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]

    device = torch.device(args.device)
    learner_args = argparse.Namespace(
        model=args.model,
        tuning=args.tuning,
        shard="ddp",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_targets,
    )
    model, tokenizer = load_model_and_tokenizer(learner_args, device)
    params = trainable_params(model)
    layout = build_layout(
        [(name, param.numel()) for name, param in params.items()],
        args.fragments,
        args.fragment_pattern,
    )
    batches = syncer_eval._probe_batches(args, tokenizer, device)
    compute_loss = lambda logits, ids, weights: sft_loss(logits, ids, args.loss_function, weights)  # noqa: E731

    buffers: dict[int, deque[list[Buffered]]] = defaultdict(
        lambda: deque(maxlen=args.buffer_rounds)
    )
    next_states = _next_state_paths(all_groups, root) if args.validate_next_state else {}
    current_state_path = None
    current_ckpt = None
    base_loss = 0.0
    base_by_batch: list[float] = []
    records = []
    was_training = model.training
    try:
        for group_idx, group in enumerate(groups, start=1):
            first = group[0]
            state_path = buffered._resolve(root, first["state_checkpoint"])
            if state_path != current_state_path:
                current_ckpt = parse_checkpoint(state_path)
                syncer_eval._apply_checkpoint(current_ckpt, layout, params, device)
                base_loss, base_by_batch = syncer_eval._losses(model, batches, compute_loss)
                current_state_path = state_path
            assert current_ckpt is not None
            fid = int(first["fragment"])
            frag = layout.fragments[fid]
            current = current_ckpt.fragments[fid][1]
            momentum = current_ckpt.fragments[fid][2]
            current_version = int(current_ckpt.fragments[fid][0])
            current_candidates = []
            current_round = []
            for row in group:
                tensor = buffered._read_f32(buffered._resolve(root, row["candidate_f32"]), frag.numel)
                candidate = buffered._candidate(row, tensor, current, momentum, current_version)
                current_candidates.append(candidate)
                current_round.append(
                    Buffered(row=row, update=candidate.update.clone(), weight=candidate.weight)
                )
            buffers[fid].append(current_round)
            if len(buffers[fid]) < args.buffer_rounds:
                continue
            history = [
                _buffered_candidate(item, current, momentum, current_version)
                for buffered_round in buffers[fid]
                for item in buffered_round
            ]
            history_rounds = [
                [
                    _buffered_candidate(item, current, momentum, current_version)
                    for item in buffered_round
                ]
                for buffered_round in buffers[fid]
            ]

            baseline_update = _production_merge_update(current_candidates, momentum, frag)
            baseline_trial = _nesterov_trial(
                current, momentum, baseline_update, args.outer_lr, args.outer_momentum
            )
            token_utility, token_se = _eval(
                model, batches, compute_loss, frag, params, current, baseline_trial,
                base_loss, base_by_batch, device,
            )
            next_state_path = next_states.get((int(first["step"]), fid))
            rel_error = None
            step_rel_error = None
            absolute_error = None
            if next_state_path is not None:
                next_state = parse_checkpoint(next_state_path).fragments[fid][1]
                absolute_error = float((baseline_trial - next_state).norm().item())
                rel_error = absolute_error / max(
                    float(next_state.norm().item()), 1e-12
                )
                step_rel_error = absolute_error / max(
                    float((next_state - current).norm().item()), 1e-12
                )
                if step_rel_error > args.max_next_state_step_relative_error:
                    raise RuntimeError(
                        "production baseline replay diverged from the captured next state: "
                        f"step={first['step']} fragment={fid} relative_error={step_rel_error:.3e} "
                        f"limit={args.max_next_state_step_relative_error:.3e}"
                    )
            out = {
                "schema": "buffered_nesterov_syncer_replay_v1",
                "seed": args.seed,
                "step": int(first["step"]),
                "syncer_global_step": int(first["syncer_global_step"]),
                "fragment": fid,
                "state_checkpoint": first["state_checkpoint"],
                "candidate_count": len(current_candidates),
                "dropped_incomplete_group_count": len(incomplete),
                "buffer_rounds": len(buffers[fid]),
                "buffer_size": len(history),
                "base_loss": base_loss,
                "token_weighted_utility": token_utility,
                "token_weighted_utility_se": token_se,
                "token_weighted_negative": token_utility < 0.0,
                "token_weighted_strict_negative": None if token_se is None else token_utility + token_se < 0.0,
                "token_weighted_selected_count": len(current_candidates),
                "token_weighted_selected_mass": 1.0,
                "token_weighted_gain_vs_token": 0.0,
                "token_weighted_next_state_absolute_error": absolute_error,
                "token_weighted_next_state_relative_error": rel_error,
                "token_weighted_next_state_step_relative_error": step_rel_error,
            }
            for policy in args.policies:
                if policy.startswith("buffer_group_"):
                    update, info = _group_policy(policy, history_rounds, momentum, frag)
                else:
                    policy_candidates = (
                        current_candidates if policy.startswith("current_") else history
                    )
                    update, info = _aggregate(policy, policy_candidates, momentum, frag, args)
                baseline_norm = float(baseline_update.norm().item())
                update_norm = float(update.norm().item())
                info["update_norm"] = update_norm
                info["baseline_update_norm"] = baseline_norm
                info["update_to_baseline_norm_ratio"] = update_norm / max(baseline_norm, 1e-12)
                info["update_cosine_to_baseline"] = soft._cosine(update, baseline_update)
                trial = _nesterov_trial(
                    current,
                    momentum,
                    update,
                    args.outer_lr * float(info.get("outer_lr_multiplier", 1.0)),
                    args.outer_momentum,
                )
                utility, utility_se = _eval(
                    model, batches, compute_loss, frag, params, current, trial,
                    base_loss, base_by_batch, device,
                )
                out[f"{policy}_utility"] = utility
                out[f"{policy}_utility_se"] = utility_se
                out[f"{policy}_negative"] = utility < 0.0
                out[f"{policy}_strict_negative"] = None if utility_se is None else utility + utility_se < 0.0
                out[f"{policy}_gain_vs_token"] = utility - token_utility
                for key, value in info.items():
                    out[f"{policy}_{key}"] = value
            records.append(out)
            if args._sink is not None:
                args._sink.write(json.dumps(_jsonable(out), sort_keys=True, allow_nan=False) + "\n")
                args._sink.flush()
            if args.progress_every and (len(records) == 1 or len(records) % args.progress_every == 0):
                print(
                    f"[buffered-nesterov] records={len(records)} groups={group_idx}/{len(groups)} "
                    f"step={out['step']} fragment={fid}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        model.train(was_training)
    return records


def summarize(records: list[dict], policies: tuple[str, ...]) -> dict:
    if not records:
        raise SystemExit("no buffered Nesterov replay records")
    token_neg = _mean([1.0 if row["token_weighted_negative"] else 0.0 for row in records])
    token_strict = _mean([
        1.0 if row["token_weighted_strict_negative"] else 0.0
        for row in records if row["token_weighted_strict_negative"] is not None
    ])
    results = {}
    for policy in policies:
        gains = [float(row[f"{policy}_gain_vs_token"]) for row in records]
        utilities = [float(row[f"{policy}_utility"]) for row in records]
        negative = _mean([1.0 if row[f"{policy}_negative"] else 0.0 for row in records])
        strict = _mean([
            1.0 if row[f"{policy}_strict_negative"] else 0.0
            for row in records if row[f"{policy}_strict_negative"] is not None
        ])
        results[policy] = {
            "mean_utility": _mean(utilities),
            "mean_gain_vs_token": _mean(gains),
            "median_gain_vs_token": _quantile(gains, 0.5),
            "gain_positive_rate": _mean([1.0 if gain > 0.0 else 0.0 for gain in gains]),
            "negative_rate": negative,
            "strict_negative_rate": strict,
            "negative_rate_relative_drop": None if token_neg <= 0.0 else (token_neg - negative) / token_neg,
            "strict_negative_rate_relative_drop": None if token_strict <= 0.0 else (token_strict - strict) / token_strict,
            "selected_mass_mean": _mean([float(row.get(f"{policy}_selected_mass", 1.0)) for row in records]),
            "normalized_effective_sample_size": _mean([
                float(row.get(f"{policy}_normalized_effective_sample_size", 1.0)) for row in records
            ]),
            "fresh_effective_share": _mean([
                float(row.get(f"{policy}_fresh_effective_share", 1.0)) for row in records
            ]),
            "history_effective_share": _mean([
                float(row.get(f"{policy}_history_effective_share", 0.0)) for row in records
            ]),
            "update_to_baseline_norm_ratio": _mean([
                float(row.get(f"{policy}_update_to_baseline_norm_ratio", 1.0)) for row in records
            ]),
            "update_cosine_to_baseline": _mean([
                float(row.get(f"{policy}_update_cosine_to_baseline", 1.0)) for row in records
            ]),
            "outer_lr_multiplier": _mean([
                float(row.get(f"{policy}_outer_lr_multiplier", 1.0)) for row in records
            ]),
            "consensus_scale": _mean([
                float(row.get(f"{policy}_consensus_scale", 1.0)) for row in records
            ]),
        }
    parameter_errors = [
        float(row["token_weighted_next_state_relative_error"])
        for row in records if row.get("token_weighted_next_state_relative_error") is not None
    ]
    step_errors = [
        float(row["token_weighted_next_state_step_relative_error"])
        for row in records if row.get("token_weighted_next_state_step_relative_error") is not None
    ]
    absolute_errors = [
        float(row["token_weighted_next_state_absolute_error"])
        for row in records if row.get("token_weighted_next_state_absolute_error") is not None
    ]
    non_token = [policy for policy in policies if policy != "token_weighted"]
    best = max(non_token, key=lambda policy: results[policy]["mean_gain_vs_token"])
    return {
        "schema": "buffered_nesterov_syncer_summary_v1",
        "records": len(records),
        "dropped_incomplete_group_count": int(records[0]["dropped_incomplete_group_count"]),
        "seeds": sorted({int(row["seed"]) for row in records if row.get("seed") is not None}),
        "buffer_rounds": int(records[0]["buffer_rounds"]),
        "buffer_size": int(records[0]["buffer_size"]),
        "outer_lr": None,
        "outer_momentum": None,
        "token_negative_rate": token_neg,
        "token_strict_negative_rate": token_strict,
        "next_state_validation": {
            "records": len(step_errors),
            "mean_absolute_error": _mean(absolute_errors),
            "max_absolute_error": max(absolute_errors) if absolute_errors else None,
            "mean_parameter_relative_error": _mean(parameter_errors),
            "max_parameter_relative_error": max(parameter_errors) if parameter_errors else None,
            "mean_step_relative_error": _mean(step_errors),
            "p95_step_relative_error": _quantile(step_errors, 0.95),
            "max_step_relative_error": max(step_errors) if step_errors else None,
        },
        "policies": results,
        "best_non_token_policy": best,
        "gate": {
            "baseline_replay_matches_next_state": bool(step_errors) and max(step_errors) < 1e-4,
            "best_mean_gain_positive": results[best]["mean_gain_vs_token"] > 0.0,
            "best_negative_drop_nonnegative": (results[best]["negative_rate_relative_drop"] or 0.0) >= 0.0,
            "best_strict_drop_nonnegative": (results[best]["strict_negative_rate_relative_drop"] or 0.0) >= 0.0,
            "best_selected_mass_ge_0.95": results[best]["selected_mass_mean"] >= 0.95,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture-dir", type=Path)
    source.add_argument("--index", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tuning", choices=["lora", "full"], default="lora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-targets", choices=["auto", "attention", "all-linear"], default="auto")
    parser.add_argument("--fragments", type=int, default=4)
    parser.add_argument("--fragment-pattern", choices=["binpack", "strided"], default="binpack")
    parser.add_argument("--loss-function", default="cross_entropy")
    parser.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    parser.add_argument("--probe-batches", type=int, default=4)
    parser.add_argument("--probe-batch-size", type=int, default=1)
    parser.add_argument("--probe-max-rows", type=int, default=128)
    parser.add_argument("--buffer-rounds", type=int, default=2)
    parser.add_argument("--history-cap", type=float, default=0.30)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--outer-momentum", type=float, default=0.9)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--expected-candidates", type=int, default=0)
    parser.add_argument("--drop-incomplete-groups", action="store_true")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--validate-next-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-next-state-step-relative-error", type=float, default=1e-4)
    parser.add_argument(
        "--policies",
        default=",".join(POLICIES[1:]),
        help="Comma-separated non-token policies to evaluate (default: all).",
    )
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    args = parser.parse_args(argv)
    requested = tuple(dict.fromkeys(part.strip() for part in args.policies.split(",") if part.strip()))
    unknown = sorted(set(requested) - set(POLICIES[1:]))
    if unknown:
        parser.error(f"unknown buffered Nesterov policies: {','.join(unknown)}")
    if not requested:
        parser.error("--policies must contain at least one non-token policy")
    if not 0.0 <= args.history_cap < 1.0:
        parser.error("--history-cap must be in [0, 1)")
    if args.buffer_rounds < 1:
        parser.error("--buffer-rounds must be >= 1")
    if args.expected_candidates < 0:
        parser.error("--expected-candidates must be >= 0")
    if args.max_next_state_step_relative_error <= 0.0:
        parser.error("--max-next-state-step-relative-error must be > 0")
    args.policies = requested
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.capture_dir or args.index.parent
    args.seed = args.seed if args.seed is not None else _infer_seed(root, args.data, args.out_jsonl)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as sink:
        args._sink = sink
        records = replay(args)
    summary = summarize(records, ("token_weighted", *args.policies))
    summary["outer_lr"] = args.outer_lr
    summary["outer_momentum"] = args.outer_momentum
    args.out_summary.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
