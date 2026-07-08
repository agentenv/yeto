#!/usr/bin/env python3
"""Replay captured syncer merge choices against a fixed probe set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_syncer_eval():
    path = REPO_ROOT / "scripts" / "evaluate_syncer_probe_capture.py"
    spec = importlib.util.spec_from_file_location("_syncer_probe_eval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


syncer_eval = _load_syncer_eval()

from yeto.export import parse_checkpoint  # noqa: E402
from yeto.fragments import build_layout  # noqa: E402
from yeto.learner import load_model_and_tokenizer, trainable_params  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402
from yeto.tensor_io import apply_fragment  # noqa: E402


@dataclass
class CandidateEval:
    row: dict
    candidate: torch.Tensor
    utility: float
    utility_se: float | None
    update_norm: float
    freshness: float
    alignment: float
    uncertainty: float
    norm_anomaly: float
    combined_score: float


def _read_index(index: Path) -> list[dict]:
    rows = []
    with index.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{index}: no capture rows")
    return rows


def _group_rows(rows: list[dict]) -> list[list[dict]]:
    groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["state_checkpoint"]), int(row["step"]), int(row["fragment"]))
        groups[key].append(row)
    return [
        sorted(group, key=lambda r: int(r["learner_id"]))
        for _, group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0]))
    ]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_f32(path: Path, expected: int) -> torch.Tensor:
    raw = path.read_bytes()
    if len(raw) != expected * 4:
        raise ValueError(f"{path}: got {len(raw)} bytes, expected {expected * 4}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _candidate_weight(candidate: CandidateEval, mode: str) -> float:
    base = float(candidate.row.get("weight", candidate.row.get("c_tokens", 1.0)))
    if mode == "token":
        return base
    if mode == "freshness":
        return base * max(candidate.freshness, 0.0)
    if mode == "hand":
        return base * max(candidate.combined_score, 0.0)
    raise ValueError(f"unknown weight mode {mode!r}")


def _weighted_candidate(
    current_flat: torch.Tensor,
    candidates: list[CandidateEval],
    *,
    mode: str,
) -> torch.Tensor:
    if not candidates:
        return current_flat.clone()
    total = 0.0
    out = torch.zeros_like(current_flat)
    for candidate in candidates:
        weight = _candidate_weight(candidate, mode)
        if weight <= 0.0 or not math.isfinite(weight):
            continue
        out.add_(candidate.candidate, alpha=weight)
        total += weight
    if total <= 0.0:
        return current_flat.clone()
    out.div_(total)
    return out


def _utility_se(base_by_batch: list[float], trial_by_batch: list[float]) -> float | None:
    utilities = [b - t for b, t in zip(base_by_batch, trial_by_batch)]
    if len(utilities) < 2:
        return None
    mean = sum(utilities) / len(utilities)
    var = sum((u - mean) ** 2 for u in utilities) / (len(utilities) - 1)
    return math.sqrt(var / len(utilities))


def _eval_flat(model, batches, compute_loss, frag, params, current_flat, trial_flat, device):
    apply_fragment(frag, trial_flat.to(device), params)
    trial_loss, trial_by_batch = syncer_eval._losses(model, batches, compute_loss)
    apply_fragment(frag, current_flat.to(device), params)
    return trial_loss, trial_by_batch


def _policy_result(
    *,
    name: str,
    selected: list[CandidateEval],
    current_flat: torch.Tensor,
    mode: str,
    outer_lr: float,
    model,
    batches,
    compute_loss,
    frag,
    params,
    base_loss: float,
    base_by_batch: list[float],
    device,
) -> dict:
    merged = _weighted_candidate(current_flat, selected, mode=mode)
    trial = current_flat + outer_lr * (merged - current_flat)
    trial_loss, trial_by_batch = _eval_flat(
        model, batches, compute_loss, frag, params, current_flat, trial, device
    )
    utility = base_loss - trial_loss
    utility_se = _utility_se(base_by_batch, trial_by_batch)
    return {
        f"{name}_utility": utility,
        f"{name}_utility_se": utility_se,
        f"{name}_negative": utility < 0.0,
        f"{name}_strict_negative": (
            None if utility_se is None else utility + utility_se < 0.0
        ),
        f"{name}_selected_count": len(selected),
    }


def _topk_count(n: int, explicit: int | None, frac: float) -> int:
    if n <= 0:
        return 0
    if explicit is not None:
        return max(1, min(n, explicit))
    return max(1, min(n, math.ceil(n * frac)))


def _quantile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return ordered[idx]


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def summarize(records: list[dict]) -> dict:
    if not records:
        raise SystemExit("no replay records")
    policies = [
        "token_weighted",
        "freshness_weighted",
        "hand_score_weighted",
        "oracle_positive",
        "oracle_topk",
        "oracle_drop_strict_bad",
        "random_positive_count",
        "random_drop_strict_bad_count",
    ]
    policy_summary = {}
    for policy in policies:
        utilities = [float(r[f"{policy}_utility"]) for r in records]
        strict_values = [
            r[f"{policy}_strict_negative"]
            for r in records
            if r[f"{policy}_strict_negative"] is not None
        ]
        policy_summary[policy] = {
            "mean_utility": sum(utilities) / len(utilities),
            "negative_merge_rate": sum(1 for u in utilities if u < 0.0) / len(utilities),
            "strict_negative_merge_rate": (
                sum(1 for v in strict_values if v) / len(strict_values)
                if strict_values
                else None
            ),
            "utility_quantiles": {
                "p05": _quantile(utilities, 0.05),
                "p50": _quantile(utilities, 0.50),
                "p95": _quantile(utilities, 0.95),
            },
            "mean_selected_count": sum(float(r[f"{policy}_selected_count"]) for r in records)
            / len(records),
        }
    headrooms = {
        "oracle_positive_minus_token": [
            float(r["oracle_positive_utility"]) - float(r["token_weighted_utility"])
            for r in records
        ],
        "oracle_topk_minus_token": [
            float(r["oracle_topk_utility"]) - float(r["token_weighted_utility"])
            for r in records
        ],
        "oracle_drop_strict_bad_minus_token": [
            float(r["oracle_drop_strict_bad_utility"]) - float(r["token_weighted_utility"])
            for r in records
        ],
    }
    return {
        "records": len(records),
        "candidate_count_mean": sum(float(r["candidate_count"]) for r in records) / len(records),
        "individual_bad_rate_mean": sum(float(r["individual_bad_rate"]) for r in records)
        / len(records),
        "individual_strict_bad_rate_mean": sum(
            float(r["individual_strict_bad_rate"]) for r in records
        )
        / len(records),
        "bad_weight_mass_mean": sum(float(r["bad_weight_mass"]) for r in records) / len(records),
        "strict_bad_weight_mass_mean": sum(
            float(r["strict_bad_weight_mass"]) for r in records
        )
        / len(records),
        "policies": policy_summary,
        "headroom": {
            name: {
                "mean": sum(values) / len(values),
                "p05": _quantile(values, 0.05),
                "p50": _quantile(values, 0.50),
                "p95": _quantile(values, 0.95),
                "positive_rate": sum(1 for v in values if v > 0.0) / len(values),
            }
            for name, values in headrooms.items()
        },
    }


def replay(args) -> list[dict]:
    root = args.capture_dir or args.index.parent
    index = args.index or root / "index.jsonl"
    rows = _read_index(index)
    groups = _group_rows(rows)
    if args.min_candidates > 1:
        groups = [group for group in groups if len(group) >= args.min_candidates]
    if args.max_groups is not None and len(groups) > args.max_groups:
        rng = random.Random(args.sample_seed)
        groups = rng.sample(groups, args.max_groups)
        groups.sort(key=lambda g: (int(g[0]["step"]), int(g[0]["fragment"])))

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
    current_base_loss = 0.0
    current_base_by_batch: list[float] = []
    norm_history: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=96))
    norm_ema: dict[int, float] = {}
    norm_var: dict[int, float] = defaultdict(lambda: 1e-4)
    rng = random.Random(args.sample_seed)
    records = []

    was_training = model.training
    try:
        for group in groups:
            first = group[0]
            state_path = _resolve(root, first["state_checkpoint"])
            if current_state_path != state_path:
                current_ckpt = parse_checkpoint(state_path)
                syncer_eval._apply_checkpoint(current_ckpt, layout, params, device)
                current_base_loss, current_base_by_batch = syncer_eval._losses(
                    model, batches, compute_loss
                )
                current_state_path = state_path
            assert current_ckpt is not None
            fid = int(first["fragment"])
            frag = layout.fragments[fid]
            current_flat = current_ckpt.fragments[fid][1]
            momentum = current_ckpt.fragments[fid][2]

            candidates: list[CandidateEval] = []
            for row in group:
                candidate = _read_f32(_resolve(root, row["candidate_f32"]), frag.numel)
                update = candidate - current_flat
                update_norm = float(update.norm().item())
                hist = norm_history[fid]
                if hist:
                    h = torch.tensor(list(hist), dtype=torch.float32)
                    med = float(h.median().item())
                    mad = float((h - med).abs().median().item()) + 1e-8
                    norm_anomaly = abs(update_norm - med) / mad
                else:
                    norm_anomaly = 0.0
                prev_mean = norm_ema.get(fid)
                if prev_mean is None:
                    uncertainty = 0.0
                else:
                    delta = update_norm - prev_mean
                    uncertainty = math.sqrt(norm_var[fid]) / (abs(prev_mean) + 1e-8)
                    norm_var[fid] = 0.85 * norm_var[fid] + 0.15 * delta * delta
                norm_ema[fid] = (
                    update_norm if prev_mean is None else 0.85 * prev_mean + 0.15 * update_norm
                )
                hist.append(update_norm)

                age = max(0, int(row["step"]) - int(row["base_version"]))
                freshness = math.exp(-age / max(args.probe_freshness_scale, 1e-9))
                alignment = _cosine(update, -momentum)
                combined_score = _sigmoid(
                    2.25 * alignment
                    + 1.35 * freshness
                    - 0.55 * math.log1p(norm_anomaly)
                    - 0.80 * uncertainty
                )
                trial = current_flat + args.probe_outer_lr * update
                trial_loss, trial_by_batch = _eval_flat(
                    model,
                    batches,
                    compute_loss,
                    frag,
                    params,
                    current_flat,
                    trial,
                    device,
                )
                candidates.append(
                    CandidateEval(
                        row=row,
                        candidate=candidate,
                        utility=current_base_loss - trial_loss,
                        utility_se=_utility_se(current_base_by_batch, trial_by_batch),
                        update_norm=update_norm,
                        freshness=freshness,
                        alignment=alignment,
                        uncertainty=uncertainty,
                        norm_anomaly=norm_anomaly,
                        combined_score=combined_score,
                    )
                )

            bad = [c.utility < 0.0 for c in candidates]
            strict_bad = [
                False if c.utility_se is None else c.utility + c.utility_se < 0.0
                for c in candidates
            ]
            weights = [max(_candidate_weight(c, "token"), 0.0) for c in candidates]
            total_weight = sum(weights)
            positive = [c for c in candidates if c.utility > 0.0]
            not_strict_bad = [c for c, is_bad in zip(candidates, strict_bad) if not is_bad]
            k = _topk_count(len(candidates), args.oracle_topk, args.oracle_topk_frac)
            topk = sorted(candidates, key=lambda c: c.utility, reverse=True)[:k]
            random_positive = rng.sample(candidates, min(len(candidates), len(positive)))
            random_not_strict = rng.sample(candidates, min(len(candidates), len(not_strict_bad)))

            out = {
                "schema": "merge_utility_replay_v1",
                "step": int(first["step"]),
                "syncer_global_step": int(first["syncer_global_step"]),
                "fragment": fid,
                "state_checkpoint": first["state_checkpoint"],
                "candidate_count": len(candidates),
                "oracle_topk_count": k,
                "base_loss": current_base_loss,
                "individual_bad_rate": sum(bad) / len(candidates),
                "individual_strict_bad_rate": sum(strict_bad) / len(candidates),
                "bad_weight_mass": (
                    sum(w for w, is_bad in zip(weights, bad) if is_bad) / total_weight
                    if total_weight > 0.0
                    else 0.0
                ),
                "strict_bad_weight_mass": (
                    sum(w for w, is_bad in zip(weights, strict_bad) if is_bad) / total_weight
                    if total_weight > 0.0
                    else 0.0
                ),
                "candidate_utility_mean": sum(c.utility for c in candidates) / len(candidates),
                "candidate_utility_min": min(c.utility for c in candidates),
                "candidate_utility_max": max(c.utility for c in candidates),
            }
            policies = [
                ("token_weighted", candidates, "token"),
                ("freshness_weighted", candidates, "freshness"),
                ("hand_score_weighted", candidates, "hand"),
                ("oracle_positive", positive, "token"),
                ("oracle_topk", topk, "token"),
                ("oracle_drop_strict_bad", not_strict_bad, "token"),
                ("random_positive_count", random_positive, "token"),
                ("random_drop_strict_bad_count", random_not_strict, "token"),
            ]
            for name, selected, mode in policies:
                out.update(
                    _policy_result(
                        name=name,
                        selected=selected,
                        current_flat=current_flat,
                        mode=mode,
                        outer_lr=args.probe_outer_lr,
                        model=model,
                        batches=batches,
                        compute_loss=compute_loss,
                        frag=frag,
                        params=params,
                        base_loss=current_base_loss,
                        base_by_batch=current_base_by_batch,
                        device=device,
                    )
                )
            records.append(out)
    finally:
        model.train(was_training)
    return records


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
    p.add_argument("--probe-batches", type=int, default=8)
    p.add_argument("--probe-batch-size", type=int, default=1)
    p.add_argument("--probe-max-rows", type=int, default=256)
    p.add_argument("--probe-outer-lr", type=float, default=1.0)
    p.add_argument("--probe-freshness-scale", type=float, default=24.0)
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--min-candidates", type=int, default=1)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--oracle-topk", type=int, default=None)
    p.add_argument("--oracle-topk-frac", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--summary-out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.capture_dir or args.index.parent
    out = args.out or root / "merge_utility_replay.jsonl"
    summary_out = args.summary_out or root / "merge_utility_summary.json"
    records = replay(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for record in records:
            f.write(json.dumps(_jsonable(record), sort_keys=True, allow_nan=False) + "\n")
    summary = summarize(records)
    text = json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    summary_out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
