#!/usr/bin/env python3
"""Replay soft robust syncer policies on captured fragment groups.

This is the EXP2.17 pivot away from brittle group-level action selection.  The
policies here never use oracle labels or anchor-probe action ranks to choose a
winner.  They transform or robustly aggregate the returned candidate fragments
and evaluate the resulting one-step update on a fixed probe set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
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


merge_replay = _load_script("replay_merge_utility")
syncer_eval = merge_replay.syncer_eval

from yeto.export import parse_checkpoint  # noqa: E402
from yeto.fragments import build_layout  # noqa: E402
from yeto.learner import load_model_and_tokenizer, trainable_params  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402
from yeto.tensor_io import apply_fragment  # noqa: E402


POLICIES = (
    "token_weighted",
    "uniform",
    "age_decay_tau4",
    "age_decay_tau8",
    "norm_clip_m1",
    "norm_clip_m15",
    "rda_clip_m1",
    "rda_clip_m15",
    "momentum_shrink_mild",
    "momentum_shrink_medium",
    "heloco_avg",
    "heloco_rda",
    "trim_norm_high1",
    "medoid",
)


@dataclass
class Candidate:
    row: dict
    tensor: torch.Tensor
    update: torch.Tensor
    weight: float
    age: float
    norm: float
    align: float


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def _group_rows(rows: list[dict], *, min_candidates: int) -> list[list[dict]]:
    groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["state_checkpoint"]), int(row["step"]), int(row["fragment"]))].append(row)
    out = [
        sorted(group, key=lambda r: int(r["learner_id"]))
        for _, group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0]))
        if len(group) >= min_candidates
    ]
    return out


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_f32(path: Path, expected: int) -> torch.Tensor:
    raw = path.read_bytes()
    if len(raw) != expected * 4:
        raise ValueError(f"{path}: got {len(raw)} bytes, expected {expected * 4}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _infer_seed(*paths: Path | None) -> int | None:
    for path in paths:
        if path is None:
            continue
        match = re.search(r"seed(\d+)", str(path))
        if match:
            return int(match.group(1))
    return None


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _quantile(values: list[float], p: float) -> float:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return float("nan")
    idx = max(0, min(len(vals) - 1, round(p * (len(vals) - 1))))
    return vals[idx]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _utility_se(base_by_batch: list[float], trial_by_batch: list[float]) -> float | None:
    utilities = [b - t for b, t in zip(base_by_batch, trial_by_batch)]
    if len(utilities) < 2:
        return None
    m = sum(utilities) / len(utilities)
    var = sum((u - m) ** 2 for u in utilities) / (len(utilities) - 1)
    return math.sqrt(var / len(utilities))


def _candidate_weight(row: dict) -> float:
    return float(row.get("weight", row.get("c_tokens", 1.0)))


def _age(row: dict) -> float:
    if row.get("current_fragment_version") is not None and row.get("base_version") is not None:
        return max(0.0, float(row["current_fragment_version"]) - float(row["base_version"]))
    return max(0.0, float(row["step"]) - float(row.get("base_version", row["step"])))


def _median(values: list[float]) -> float:
    return _quantile(values, 0.5)


def _weights(candidates: list[Candidate], policy: str) -> list[float]:
    if policy == "uniform":
        return [1.0 for _ in candidates]
    if policy == "age_decay_tau4":
        return [c.weight * math.exp(-c.age / 4.0) for c in candidates]
    if policy == "age_decay_tau8":
        return [c.weight * math.exp(-c.age / 8.0) for c in candidates]
    return [c.weight for c in candidates]


def _weighted_update(candidates: list[Candidate], weights: list[float], updates: list[torch.Tensor]) -> torch.Tensor:
    total = sum(w for w in weights if w > 0.0 and math.isfinite(w))
    if total <= 0.0:
        return torch.zeros_like(updates[0])
    out = torch.zeros_like(updates[0])
    for update, weight in zip(updates, weights):
        if weight > 0.0 and math.isfinite(weight):
            out.add_(update, alpha=weight)
    return out.div(total)


def _clip_updates(updates: list[torch.Tensor], norms: list[float], multiplier: float) -> tuple[list[torch.Tensor], float, float]:
    clip = max(_median(norms) * multiplier, 1e-12)
    clipped = []
    scales = []
    for update, norm in zip(updates, norms):
        scale = min(1.0, clip / max(norm, 1e-12))
        clipped.append(update * scale)
        scales.append(scale)
    return clipped, clip, _mean(scales)


def _rda_update(candidates: list[Candidate], weights: list[float], updates: list[torch.Tensor]) -> torch.Tensor:
    total = sum(w for w in weights if w > 0.0 and math.isfinite(w))
    if total <= 0.0:
        return torch.zeros_like(updates[0])
    norms = [float(u.norm().item()) for u in updates]
    radial = sum(w * n for w, n in zip(weights, norms) if w > 0.0 and math.isfinite(w)) / total
    direction = torch.zeros_like(updates[0])
    for update, weight, norm in zip(updates, weights, norms):
        if weight > 0.0 and math.isfinite(weight) and norm > 1e-12:
            direction.add_(update, alpha=weight / total / norm)
    dnorm = float(direction.norm().item())
    if dnorm < 1e-12:
        return _weighted_update(candidates, weights, updates)
    return direction.mul(radial / dnorm)


def _heloco_update(update: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
    # Rust syncer correction operates on delta = current - candidate and
    # momentum. Here update = candidate - current, so convert signs around it.
    delta = -update.clone()
    du = float(delta.norm().item())
    dm = float(momentum.norm().item())
    eps = 1e-8
    if du < eps or dm < eps:
        return update
    c = float(torch.dot(delta, momentum).item() / (du * dm))
    c_ok, k_s, k_d, beta_max, kappa = 0.2, 0.5, 1.0, 0.5, 3.0
    if c >= c_ok:
        return update
    conf = du / (du + kappa * dm + eps)
    if c < 0.0:
        beta = min(k_s * (-c) * conf, beta_max)
        delta.add_(momentum, alpha=(-beta * c * du / dm))
    else:
        lam = min(k_d * (1.0 - c) * conf, 1.0)
        corrected = delta * ((1.0 - lam) / du) + momentum * (lam / dm)
        cnorm = float(corrected.norm().item())
        if cnorm >= eps:
            delta = corrected.mul(du / cnorm)
    return -delta


def _aggregate(policy: str, current_flat: torch.Tensor, candidates: list[Candidate], momentum: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if not candidates:
        return current_flat.clone(), {"effective_count": 0, "selected_mass": 0.0}
    selected = list(candidates)
    updates = [c.update for c in selected]
    weights = _weights(selected, policy)
    info = {
        "selected_count": len(selected),
        "selected_mass": 1.0,
        "effective_count": len(selected),
        "clip_norm": None,
        "mean_clip_scale": 1.0,
        "mean_age": _mean([c.age for c in selected]),
        "mean_alignment": _mean([c.align for c in selected]),
    }
    if policy == "trim_norm_high1" and len(selected) >= 3:
        drop_idx = max(range(len(selected)), key=lambda i: selected[i].norm)
        total = sum(max(c.weight, 0.0) for c in selected)
        selected = [c for i, c in enumerate(selected) if i != drop_idx]
        updates = [c.update for c in selected]
        weights = [c.weight for c in selected]
        kept = sum(max(c.weight, 0.0) for c in selected)
        info["selected_count"] = len(selected)
        info["selected_mass"] = kept / total if total > 0.0 else 0.0
        merged_update = _weighted_update(selected, weights, updates)
        return current_flat + merged_update, info
    if policy == "medoid":
        best_idx = 0
        best_dist = float("inf")
        for i, update_i in enumerate(updates):
            dist = 0.0
            for update_j, weight_j in zip(updates, weights):
                dist += max(weight_j, 0.0) * float((update_i - update_j).norm().item())
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        info["selected_count"] = 1
        total = sum(max(c.weight, 0.0) for c in selected)
        info["selected_mass"] = max(selected[best_idx].weight, 0.0) / total if total > 0.0 else 0.0
        return current_flat + updates[best_idx], info
    if policy in {"norm_clip_m1", "rda_clip_m1"}:
        updates, clip, scale = _clip_updates(updates, [c.norm for c in selected], 1.0)
        info["clip_norm"] = clip
        info["mean_clip_scale"] = scale
    if policy in {"norm_clip_m15", "rda_clip_m15"}:
        updates, clip, scale = _clip_updates(updates, [c.norm for c in selected], 1.5)
        info["clip_norm"] = clip
        info["mean_clip_scale"] = scale
    if policy == "momentum_shrink_mild":
        updates = [u * (1.0 - 0.25 * max(0.0, -c.align)) for u, c in zip(updates, selected)]
    if policy == "momentum_shrink_medium":
        updates = [u * (1.0 - 0.50 * max(0.0, -c.align)) for u, c in zip(updates, selected)]
    if policy in {"heloco_avg", "heloco_rda"}:
        updates = [_heloco_update(u, momentum) for u in updates]
    if policy in {"rda_clip_m1", "rda_clip_m15", "heloco_rda"}:
        merged_update = _rda_update(selected, weights, updates)
    else:
        merged_update = _weighted_update(selected, weights, updates)
    return current_flat + merged_update, info


def _eval_trial(model, batches, compute_loss, frag, params, current_flat, trial_flat, base_loss, base_by_batch, device):
    apply_fragment(frag, trial_flat.to(device), params)
    trial_loss, trial_by_batch = syncer_eval._losses(model, batches, compute_loss)
    apply_fragment(frag, current_flat.to(device), params)
    utility = base_loss - trial_loss
    utility_se = _utility_se(base_by_batch, trial_by_batch)
    return utility, utility_se


def replay(args) -> list[dict]:
    root = args.capture_dir or args.index.parent
    index = args.index or root / "index.jsonl"
    rows = _read_jsonl(index)
    groups = _group_rows(rows, min_candidates=args.min_candidates)
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
        [(n, p.numel()) for n, p in params.items()], args.fragments, args.fragment_pattern
    )
    batches = syncer_eval._probe_batches(args, tokenizer, device)
    compute_loss = lambda logits, ids, w: sft_loss(logits, ids, args.loss_function, w)  # noqa: E731

    current_state_path: Path | None = None
    current_ckpt = None
    base_loss = 0.0
    base_by_batch: list[float] = []
    records = []
    was_training = model.training
    try:
        for idx, group in enumerate(groups, start=1):
            first = group[0]
            state_path = _resolve(root, first["state_checkpoint"])
            if state_path != current_state_path:
                current_ckpt = parse_checkpoint(state_path)
                syncer_eval._apply_checkpoint(current_ckpt, layout, params, device)
                base_loss, base_by_batch = syncer_eval._losses(model, batches, compute_loss)
                current_state_path = state_path
            assert current_ckpt is not None
            fid = int(first["fragment"])
            frag = layout.fragments[fid]
            current_flat = current_ckpt.fragments[fid][1]
            momentum = current_ckpt.fragments[fid][2]
            ref = -momentum
            candidates = []
            for row in group:
                tensor = _read_f32(_resolve(root, row["candidate_f32"]), frag.numel)
                update = tensor - current_flat
                candidates.append(
                    Candidate(
                        row=row,
                        tensor=tensor,
                        update=update,
                        weight=_candidate_weight(row),
                        age=_age(row),
                        norm=float(update.norm().item()),
                        align=_cosine(update, ref),
                    )
                )
            out = {
                "schema": "soft_robust_syncer_replay_v1",
                "seed": args.seed,
                "step": int(first["step"]),
                "syncer_global_step": int(first["syncer_global_step"]),
                "fragment": fid,
                "state_checkpoint": first["state_checkpoint"],
                "candidate_count": len(candidates),
                "base_loss": base_loss,
                "candidate_norm_mean": _mean([c.norm for c in candidates]),
                "candidate_norm_p50": _median([c.norm for c in candidates]),
                "candidate_age_mean": _mean([c.age for c in candidates]),
                "candidate_alignment_mean": _mean([c.align for c in candidates]),
            }
            for policy in POLICIES:
                trial, info = _aggregate(policy, current_flat, candidates, momentum)
                utility, utility_se = _eval_trial(
                    model,
                    batches,
                    compute_loss,
                    frag,
                    params,
                    current_flat,
                    trial,
                    base_loss,
                    base_by_batch,
                    device,
                )
                prefix = policy
                out[f"{prefix}_utility"] = utility
                out[f"{prefix}_utility_se"] = utility_se
                out[f"{prefix}_negative"] = utility < 0.0
                out[f"{prefix}_strict_negative"] = None if utility_se is None else utility + utility_se < 0.0
                for key, value in info.items():
                    out[f"{prefix}_{key}"] = value
                out[f"{prefix}_gain_vs_token"] = None
            token = float(out["token_weighted_utility"])
            for policy in POLICIES:
                out[f"{policy}_gain_vs_token"] = float(out[f"{policy}_utility"]) - token
            records.append(out)
            sink = getattr(args, "_record_sink", None)
            if sink is not None:
                sink.write(json.dumps(_jsonable(out), sort_keys=True, allow_nan=False) + "\n")
                sink.flush()
            if args.progress_every and (idx == 1 or idx % args.progress_every == 0 or idx == len(groups)):
                print(f"[soft-robust] {idx}/{len(groups)} groups step={out['step']} fragment={fid}", file=sys.stderr, flush=True)
    finally:
        model.train(was_training)
    return records


def summarize(records: list[dict]) -> dict:
    if not records:
        raise SystemExit("no records")
    token_neg = _mean([1.0 if r["token_weighted_negative"] else 0.0 for r in records])
    token_strict = _mean(
        [
            1.0 if r["token_weighted_strict_negative"] else 0.0
            for r in records
            if r["token_weighted_strict_negative"] is not None
        ]
    )
    policies = {}
    for policy in POLICIES:
        utils = [float(r[f"{policy}_utility"]) for r in records]
        gains = [float(r[f"{policy}_gain_vs_token"]) for r in records]
        neg = _mean([1.0 if r[f"{policy}_negative"] else 0.0 for r in records])
        strict = _mean(
            [
                1.0 if r[f"{policy}_strict_negative"] else 0.0
                for r in records
                if r[f"{policy}_strict_negative"] is not None
            ]
        )
        policies[policy] = {
            "mean_utility": _mean(utils),
            "mean_gain_vs_token": _mean(gains),
            "median_gain_vs_token": _quantile(gains, 0.5),
            "gain_positive_rate": _mean([1.0 if g > 0.0 else 0.0 for g in gains]),
            "gain_quantiles": {"p05": _quantile(gains, 0.05), "p50": _quantile(gains, 0.5), "p95": _quantile(gains, 0.95)},
            "negative_rate": neg,
            "strict_negative_rate": strict,
            "negative_rate_relative_drop": None if token_neg <= 0.0 else (token_neg - neg) / token_neg,
            "strict_negative_rate_relative_drop": None if token_strict <= 0.0 else (token_strict - strict) / token_strict,
            "selected_mass_mean": _mean([float(r.get(f"{policy}_selected_mass", 1.0)) for r in records]),
            "selected_count_mean": _mean([float(r.get(f"{policy}_selected_count", r["candidate_count"])) for r in records]),
            "mean_clip_scale": _mean([float(r.get(f"{policy}_mean_clip_scale", 1.0)) for r in records]),
        }
    best = max((p for p in POLICIES if p != "token_weighted"), key=lambda p: policies[p]["mean_gain_vs_token"])
    return {
        "schema": "soft_robust_syncer_summary_v1",
        "records": len(records),
        "seeds": sorted({int(r["seed"]) for r in records if r.get("seed") is not None}),
        "candidate_count_mean": _mean([float(r["candidate_count"]) for r in records]),
        "token_negative_rate": token_neg,
        "token_strict_negative_rate": token_strict,
        "policies": policies,
        "best_non_token_policy": best,
        "gate": {
            "best_mean_gain_positive": policies[best]["mean_gain_vs_token"] > 0.0,
            "best_negative_drop_positive": (policies[best]["negative_rate_relative_drop"] or 0.0) > 0.0,
            "best_strict_drop_positive": (policies[best]["strict_negative_rate_relative_drop"] or 0.0) > 0.0,
            "best_selected_mass_ge_0.95": policies[best]["selected_mass_mean"] >= 0.95,
        },
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture-dir", type=Path)
    src.add_argument("--index", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tuning", choices=["lora", "full"], default="lora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-targets", choices=["auto", "attention", "all-linear"], default="auto")
    p.add_argument("--fragments", type=int, default=4)
    p.add_argument("--fragment-pattern", choices=["binpack", "strided"], default="binpack")
    p.add_argument("--loss-function", default="cross_entropy")
    p.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    p.add_argument("--probe-batches", type=int, default=4)
    p.add_argument("--probe-batch-size", type=int, default=1)
    p.add_argument("--probe-max-rows", type=int, default=128)
    p.add_argument("--min-candidates", type=int, default=2)
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--out-jsonl", required=True, type=Path)
    p.add_argument("--out-summary", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.capture_dir or args.index.parent
    args.seed = args.seed if args.seed is not None else _infer_seed(root, args.data, args.out_jsonl)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as f:
        args._record_sink = f
        records = replay(args)
    summary = summarize(records)
    args.out_summary.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
