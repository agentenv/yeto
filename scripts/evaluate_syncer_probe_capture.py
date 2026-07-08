#!/usr/bin/env python3
"""Evaluate syncer-current fragment utility from captured candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from yeto.data import build_packed_dataset
from yeto.export import parse_checkpoint, validate_against_layout
from yeto.fragments import build_layout
from yeto.learner import load_model_and_tokenizer, trainable_params
from yeto.losses import sft_loss
from yeto.tensor_io import apply_fragment


def _load_probe_summarizer():
    path = REPO_ROOT / "scripts" / "summarize_fragment_probe.py"
    spec = importlib.util.spec_from_file_location("_syncer_probe_summary", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _read_index(index: Path) -> list[dict]:
    rows = []
    with index.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{index}: no capture records")
    return rows


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


def _probe_batches(args, tokenizer, device):
    ds = build_packed_dataset(
        str(args.data),
        tokenizer,
        learner_id=0,
        num_learners=1,
        seq_len=args.seq_len,
        max_rows=args.probe_max_rows,
        train_on=args.train_on,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.probe_batch_size, shuffle=False, drop_last=False
    )
    batches = []
    for input_ids, weights in loader:
        batches.append(
            (input_ids.to(device, non_blocking=True), weights.to(device, non_blocking=True))
        )
        if len(batches) >= args.probe_batches:
            break
    if not batches:
        raise SystemExit("--data produced no probe batches")
    return batches


def _losses(model, batches, compute_loss) -> tuple[float, list[float]]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0
    by_batch = []
    with torch.no_grad():
        for input_ids, weights in batches:
            out = model(input_ids=input_ids)
            loss, n = compute_loss(out.logits, input_ids, weights)
            loss_value = float(loss.item())
            token_count = float(n.item())
            total_loss += loss_value
            total_tokens += token_count
            by_batch.append(loss_value / max(token_count, 1.0))
    return total_loss / max(total_tokens, 1.0), by_batch


def _utility_se(base_by_batch: list[float], trial_by_batch: list[float]) -> float | None:
    utilities = [b - t for b, t in zip(base_by_batch, trial_by_batch)]
    if len(utilities) < 2:
        return None
    mean = sum(utilities) / len(utilities)
    var = sum((u - mean) ** 2 for u in utilities) / (len(utilities) - 1)
    return math.sqrt(var / len(utilities))


def _apply_checkpoint(ckpt, layout, params, device) -> None:
    validate_against_layout(ckpt, layout)
    for frag, (_, flat_params, _) in zip(layout.fragments, ckpt.fragments):
        apply_fragment(frag, flat_params.to(device), params)


def evaluate(args) -> list[dict]:
    root = args.capture_dir or args.index.parent
    index = args.index or root / "index.jsonl"
    rows = _read_index(index)
    if args.max_records is not None and len(rows) > args.max_records:
        rng = random.Random(args.sample_seed)
        rows = rng.sample(rows, args.max_records)
    rows.sort(key=lambda r: (str(r["state_checkpoint"]), int(r["step"]), int(r["learner_id"])))

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
    batches = _probe_batches(args, tokenizer, device)
    compute_loss = lambda logits, ids, w: sft_loss(logits, ids, args.loss_function, w)  # noqa: E731

    current_state_path: Path | None = None
    current_ckpt = None
    current_base_loss = 0.0
    current_base_by_batch: list[float] = []
    norm_history: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=96))
    norm_ema: dict[int, float] = {}
    norm_var: dict[int, float] = defaultdict(lambda: 1e-4)
    records = []
    was_training = model.training
    try:
        for row in rows:
            state_path = _resolve(root, row["state_checkpoint"])
            if current_state_path != state_path:
                current_ckpt = parse_checkpoint(state_path)
                _apply_checkpoint(current_ckpt, layout, params, device)
                current_base_loss, current_base_by_batch = _losses(model, batches, compute_loss)
                current_state_path = state_path
            assert current_ckpt is not None
            fid = int(row["fragment"])
            frag = layout.fragments[fid]
            current_flat = current_ckpt.fragments[fid][1]
            momentum = current_ckpt.fragments[fid][2]
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
            norm_ema[fid] = update_norm if prev_mean is None else 0.85 * prev_mean + 0.15 * update_norm
            hist.append(update_norm)

            age = max(0, int(row["step"]) - int(row["base_version"]))
            freshness = math.exp(-age / max(args.probe_freshness_scale, 1e-9))
            alignment = _cosine(update, -momentum)
            combined_logit = (
                2.25 * alignment
                + 1.35 * freshness
                - 0.55 * math.log1p(norm_anomaly)
                - 0.80 * uncertainty
            )
            combined_score = _sigmoid(combined_logit)

            trial = current_flat + args.probe_outer_lr * update
            apply_fragment(frag, trial.to(device), params)
            trial_loss, trial_by_batch = _losses(model, batches, compute_loss)
            apply_fragment(frag, current_flat.to(device), params)
            utility = current_base_loss - trial_loss
            utility_se = _utility_se(current_base_by_batch, trial_by_batch)
            out = {
                "schema": "fragment_probe_v2",
                "oracle_scope": "syncer_current_global",
                "learner_id": int(row["learner_id"]),
                "fragment": fid,
                "pull_step": int(row["step"]),
                "base_version": int(row["base_version"]),
                "syncer_global_step": int(row["syncer_global_step"]),
                "current_fragment_version": int(row["current_fragment_version"]),
                "local_step": int(row["local_step"]),
                "c_steps": int(row["c_steps"]),
                "c_tokens": int(row["c_tokens"]),
                "age": age,
                "freshness": freshness,
                "alignment": alignment,
                "uncertainty": uncertainty,
                "norm_anomaly": norm_anomaly,
                "combined_score": combined_score,
                "update_norm": update_norm,
                "base_loss": current_base_loss,
                "trial_loss": trial_loss,
                "utility": utility,
                "utility_se": utility_se,
                "bad_strict": None if utility_se is None else utility + utility_se < 0.0,
                "probe_batches": len(batches),
                "probe_outer_lr": args.probe_outer_lr,
                "capture_state_checkpoint": row["state_checkpoint"],
                "capture_candidate_f32": row["candidate_f32"],
            }
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
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--summary-out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.capture_dir or args.index.parent
    out = args.out or root / "syncer_current_probe.jsonl"
    summary_out = args.summary_out or root / "syncer_current_probe_summary.json"
    records = evaluate(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    summarizer = _load_probe_summarizer()
    summary = summarizer.summarize([out])
    text = json.dumps(summarizer.jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    summary_out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
