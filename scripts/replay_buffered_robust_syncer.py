#!/usr/bin/env python3
"""Replay sliding-buffer robust merges on captured syncer fragments.

The captured trajectory still comes from the original token-weighted run, so
this is a one-step screen rather than an exact counterfactual trajectory.  At
each captured state, the baseline uses the current group while buffered
policies use the most recent B candidates for the same fragment.
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


soft = _load_script("replay_soft_robust_syncer")
syncer_eval = soft.syncer_eval

from yeto.export import parse_checkpoint  # noqa: E402
from yeto.fragments import build_layout  # noqa: E402
from yeto.learner import load_model_and_tokenizer, trainable_params  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402
from yeto.tensor_io import apply_fragment  # noqa: E402


POLICIES = (
    "token_weighted",
    "buffer_token",
    "buffer_norm_clip_m1",
    "buffer_norm_clip_m15",
    "buffer_rda_clip_m1",
    "buffer_age4_clip_m1",
    "buffer_consensus_a1",
    "buffer_consensus_a2",
    "buffer_geomedian",
    "buffer_coord_median",
    "buffer_coord_median_blend25",
    "buffer_coord_median_blend50",
    "buffer_coord_trim1",
    "buffer_heloco",
)


@dataclass
class Buffered:
    row: dict
    tensor: torch.Tensor


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


def _group_rows(rows: list[dict], min_candidates: int) -> list[list[dict]]:
    groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["state_checkpoint"]), int(row["step"]), int(row["fragment"]))].append(row)
    return [
        sorted(group, key=lambda r: int(r["learner_id"]))
        for _, group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0]))
        if len(group) >= min_candidates
    ]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_f32(path: Path, expected: int) -> torch.Tensor:
    raw = path.read_bytes()
    if len(raw) != expected * 4:
        raise ValueError(f"{path}: got {len(raw)} bytes, expected {expected * 4}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _quantile(values: list[float], p: float) -> float:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return float("nan")
    return vals[max(0, min(len(vals) - 1, round(p * (len(vals) - 1))))]


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


def _utility_se(base_by_batch: list[float], trial_by_batch: list[float]) -> float | None:
    values = [base - trial for base, trial in zip(base_by_batch, trial_by_batch)]
    if len(values) < 2:
        return None
    avg = sum(values) / len(values)
    var = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var / len(values))


def _candidate(
    row: dict,
    tensor: torch.Tensor,
    current: torch.Tensor,
    momentum: torch.Tensor,
    current_version: int,
) -> soft.Candidate:
    update = tensor - current
    return soft.Candidate(
        row=row,
        tensor=tensor,
        update=update,
        weight=soft._candidate_weight(row),
        age=max(0.0, float(current_version) - float(row.get("base_version", current_version))),
        norm=float(update.norm().item()),
        align=soft._cosine(update, -momentum),
    )


def _weighted(updates: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    total = sum(max(weight, 0.0) for weight in weights)
    if total <= 0.0:
        return torch.zeros_like(updates[0])
    out = torch.zeros_like(updates[0])
    for update, weight in zip(updates, weights):
        if weight > 0.0 and math.isfinite(weight):
            out.add_(update, alpha=weight)
    return out.div(total)


def _geomedian(updates: list[torch.Tensor], weights: list[float], iterations: int = 5) -> torch.Tensor:
    center = _weighted(updates, weights)
    for _ in range(iterations):
        distances = [max(float((update - center).norm().item()), 1e-8) for update in updates]
        effective = [max(weight, 0.0) / distance for weight, distance in zip(weights, distances)]
        center = _weighted(updates, effective)
    return center


def _aggregate(policy: str, candidates: list[soft.Candidate], momentum: torch.Tensor) -> tuple[torch.Tensor, dict]:
    updates = [candidate.update for candidate in candidates]
    weights = [candidate.weight for candidate in candidates]
    norms = [candidate.norm for candidate in candidates]
    info = {
        "selected_count": len(candidates),
        "selected_mass": 1.0,
        "mean_age": _mean([candidate.age for candidate in candidates]),
        "p95_age": _quantile([candidate.age for candidate in candidates], 0.95),
        "mean_clip_scale": 1.0,
    }
    if policy == "buffer_token":
        return _weighted(updates, weights), info
    if policy in {"buffer_norm_clip_m1", "buffer_rda_clip_m1", "buffer_age4_clip_m1"}:
        updates, _, scale = soft._clip_updates(updates, norms, 1.0)
        info["mean_clip_scale"] = scale
    elif policy == "buffer_norm_clip_m15":
        updates, _, scale = soft._clip_updates(updates, norms, 1.5)
        info["mean_clip_scale"] = scale
    if policy == "buffer_age4_clip_m1":
        weights = [candidate.weight * math.exp(-candidate.age / 4.0) for candidate in candidates]
    if policy in {"buffer_consensus_a1", "buffer_consensus_a2"}:
        provisional = _weighted(updates, weights)
        alpha = 1.0 if policy.endswith("a1") else 2.0
        agreement = [max(-1.0, min(1.0, soft._cosine(update, provisional))) for update in updates]
        weights = [weight * math.exp(-alpha * (1.0 - cosine)) for weight, cosine in zip(weights, agreement)]
    if policy == "buffer_geomedian":
        return _geomedian(updates, weights), info
    if policy in {"buffer_coord_median", "buffer_coord_median_blend25", "buffer_coord_median_blend50"}:
        median = torch.stack(updates).median(dim=0).values
        if policy == "buffer_coord_median":
            return median, info
        token = _weighted(updates, weights)
        blend = 0.25 if policy.endswith("blend25") else 0.50
        return token.mul(1.0 - blend).add(median, alpha=blend), info
    if policy == "buffer_coord_trim1" and len(updates) >= 5:
        stacked = torch.stack(updates).sort(dim=0).values
        return stacked[1:-1].mean(dim=0), info
    if policy == "buffer_heloco":
        updates = [soft._heloco_update(update, momentum) for update in updates]
    if policy == "buffer_rda_clip_m1":
        return soft._rda_update(candidates, weights, updates), info
    return _weighted(updates, weights), info


def _eval(model, batches, compute_loss, frag, params, current, trial, base_loss, base_by_batch, device):
    apply_fragment(frag, trial.to(device), params)
    trial_loss, trial_by_batch = syncer_eval._losses(model, batches, compute_loss)
    apply_fragment(frag, current.to(device), params)
    utility = base_loss - trial_loss
    utility_se = _utility_se(base_by_batch, trial_by_batch)
    return utility, utility_se


def replay(args) -> list[dict]:
    root = args.capture_dir or args.index.parent
    index = args.index or root / "index.jsonl"
    groups = _group_rows(_read_jsonl(index), args.min_candidates)
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
    layout = build_layout([(name, param.numel()) for name, param in params.items()], args.fragments, args.fragment_pattern)
    batches = syncer_eval._probe_batches(args, tokenizer, device)
    compute_loss = lambda logits, ids, weights: sft_loss(logits, ids, args.loss_function, weights)  # noqa: E731

    buffers: dict[int, deque[Buffered]] = defaultdict(lambda: deque(maxlen=args.buffer_size))
    current_state_path = None
    current_ckpt = None
    base_loss = 0.0
    base_by_batch: list[float] = []
    records = []
    was_training = model.training
    try:
        for group_idx, group in enumerate(groups, start=1):
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
            current = current_ckpt.fragments[fid][1]
            momentum = current_ckpt.fragments[fid][2]
            current_version = int(current_ckpt.fragments[fid][0])
            current_candidates = []
            for row in group:
                tensor = _read_f32(_resolve(root, row["candidate_f32"]), frag.numel)
                current_candidates.append(_candidate(row, tensor, current, momentum, current_version))
                buffers[fid].append(Buffered(row=row, tensor=tensor))
            if len(buffers[fid]) < args.buffer_size:
                continue
            buffered = [
                _candidate(item.row, item.tensor, current, momentum, current_version)
                for item in buffers[fid]
            ]
            baseline_update, baseline_info = _aggregate("buffer_token", current_candidates, momentum)
            token_utility, token_se = _eval(
                model, batches, compute_loss, frag, params, current, current + baseline_update,
                base_loss, base_by_batch, device,
            )
            out = {
                "schema": "buffered_robust_syncer_replay_v1",
                "seed": args.seed,
                "step": int(first["step"]),
                "syncer_global_step": int(first["syncer_global_step"]),
                "fragment": fid,
                "state_checkpoint": first["state_checkpoint"],
                "candidate_count": len(current_candidates),
                "buffer_size": len(buffered),
                "base_loss": base_loss,
                "token_weighted_utility": token_utility,
                "token_weighted_utility_se": token_se,
                "token_weighted_negative": token_utility < 0.0,
                "token_weighted_strict_negative": None if token_se is None else token_utility + token_se < 0.0,
                "token_weighted_selected_count": baseline_info["selected_count"],
                "token_weighted_selected_mass": 1.0,
                "token_weighted_gain_vs_token": 0.0,
            }
            for policy in POLICIES[1:]:
                update, info = _aggregate(policy, buffered, momentum)
                utility, utility_se = _eval(
                    model, batches, compute_loss, frag, params, current, current + update,
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
            sink = getattr(args, "_sink", None)
            if sink is not None:
                sink.write(json.dumps(_jsonable(out), sort_keys=True, allow_nan=False) + "\n")
                sink.flush()
            if args.progress_every and (len(records) == 1 or len(records) % args.progress_every == 0):
                print(f"[buffered] records={len(records)} groups={group_idx}/{len(groups)} step={out['step']} fragment={fid}", file=sys.stderr, flush=True)
    finally:
        model.train(was_training)
    return records


def summarize(records: list[dict]) -> dict:
    if not records:
        raise SystemExit("no buffered replay records")
    token_neg = _mean([1.0 if row["token_weighted_negative"] else 0.0 for row in records])
    token_strict = _mean([
        1.0 if row["token_weighted_strict_negative"] else 0.0
        for row in records if row["token_weighted_strict_negative"] is not None
    ])
    policies = {}
    for policy in POLICIES:
        gains = [float(row[f"{policy}_gain_vs_token"]) for row in records]
        utilities = [float(row[f"{policy}_utility"]) for row in records]
        neg = _mean([1.0 if row[f"{policy}_negative"] else 0.0 for row in records])
        strict = _mean([
            1.0 if row[f"{policy}_strict_negative"] else 0.0
            for row in records if row[f"{policy}_strict_negative"] is not None
        ])
        policies[policy] = {
            "mean_utility": _mean(utilities),
            "mean_gain_vs_token": _mean(gains),
            "median_gain_vs_token": _quantile(gains, 0.5),
            "gain_positive_rate": _mean([1.0 if gain > 0.0 else 0.0 for gain in gains]),
            "negative_rate": neg,
            "strict_negative_rate": strict,
            "negative_rate_relative_drop": None if token_neg <= 0.0 else (token_neg - neg) / token_neg,
            "strict_negative_rate_relative_drop": None if token_strict <= 0.0 else (token_strict - strict) / token_strict,
            "selected_mass_mean": _mean([float(row.get(f"{policy}_selected_mass", 1.0)) for row in records]),
            "mean_age": _mean([float(row.get(f"{policy}_mean_age", 0.0)) for row in records]),
            "p95_age": _mean([float(row.get(f"{policy}_p95_age", 0.0)) for row in records]),
        }
    best = max(POLICIES[1:], key=lambda policy: policies[policy]["mean_gain_vs_token"])
    return {
        "schema": "buffered_robust_syncer_summary_v1",
        "records": len(records),
        "seeds": sorted({int(row["seed"]) for row in records if row.get("seed") is not None}),
        "buffer_size": int(records[0]["buffer_size"]),
        "token_negative_rate": token_neg,
        "token_strict_negative_rate": token_strict,
        "policies": policies,
        "best_non_token_policy": best,
        "gate": {
            "best_mean_gain_positive": policies[best]["mean_gain_vs_token"] > 0.0,
            "best_negative_drop_positive": (policies[best]["negative_rate_relative_drop"] or 0.0) > 0.0,
            "best_strict_drop_nonnegative": (policies[best]["strict_negative_rate_relative_drop"] or 0.0) >= 0.0,
            "best_selected_mass_ge_0.95": policies[best]["selected_mass_mean"] >= 0.95,
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
    parser.add_argument("--buffer-size", type=int, default=8)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.capture_dir or args.index.parent
    args.seed = args.seed if args.seed is not None else _infer_seed(root, args.data, args.out_jsonl)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as sink:
        args._sink = sink
        records = replay(args)
    summary = summarize(records)
    args.out_summary.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
