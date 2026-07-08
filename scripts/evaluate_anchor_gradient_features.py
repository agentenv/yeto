#!/usr/bin/env python3
"""Add syncer-current anchor-gradient features to captured candidate records."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from collections import defaultdict
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
from yeto.data import build_packed_dataset, load_rows  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path}: no records")
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n")


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


def _fragment_flat_from_grads(frag, params: dict[str, torch.Tensor]) -> torch.Tensor:
    pieces = []
    for name, _ in frag.tensors:
        grad = params[name].grad
        pieces.append(torch.zeros_like(params[name]).reshape(-1).float() if grad is None else grad.detach().reshape(-1).float())
    return torch.cat(pieces).cpu()


def _anchor_gradient(model, batches, compute_loss, *, mean_loss: bool = True) -> tuple[float, float]:
    model.zero_grad(set_to_none=True)
    total_loss = None
    total_tokens = 0.0
    for input_ids, weights in batches:
        out = model(input_ids=input_ids)
        loss, n = compute_loss(out.logits, input_ids, weights)
        total_loss = loss if total_loss is None else total_loss + loss
        total_tokens += float(n.detach().item())
    if total_loss is None:
        raise RuntimeError("no anchor batches")
    denom = max(total_tokens, 1.0) if mean_loss else 1.0
    objective = total_loss / denom
    objective.backward()
    return float(objective.detach().item()), total_tokens


def _read_capture_index(capture_dir: Path) -> dict[tuple[str, int, int], list[dict]]:
    index = capture_dir / "index.jsonl"
    if not index.exists():
        return {}
    groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in _read_jsonl(index):
        key = (str(row["state_checkpoint"]), int(row["step"]), int(row["fragment"]))
        groups[key].append(row)
    return groups


def _read_capture_rows(capture_dir: Path) -> list[dict]:
    index = capture_dir / "index.jsonl"
    if not index.exists():
        raise SystemExit(f"{index}: missing capture index")
    return _read_jsonl(index)


def _plain_row(row) -> dict:
    return json.loads(json.dumps(dict(row), sort_keys=True))


def _load_plain_rows(path: Path | str) -> list[dict]:
    rows = load_rows(str(path))
    return [_plain_row(rows[i]) for i in range(len(rows))]


def _row_hash(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_order(rows: list[dict], *, salt: str) -> list[int]:
    def key(i: int) -> str:
        payload = json.dumps(rows[i], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(f"{salt}\n{i}\n{payload}".encode("utf-8")).hexdigest()

    return sorted(range(len(rows)), key=key)


def _select_rows(rows: list[dict], *, max_rows: int, salt: str) -> tuple[list[int], list[dict]]:
    order = _stable_order(rows, salt=salt)
    ids = order[: min(max_rows, len(order))]
    return ids, [rows[i] for i in ids]


def _combined_hash(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(_row_hash(row).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def _data_splits(args) -> tuple[list[dict], list[dict], dict]:
    data = args.data
    anchor_data = args.anchor_data
    oracle_data = args.oracle_data
    if data is None and (anchor_data is None or oracle_data is None):
        raise SystemExit("provide --data, or both --anchor-data and --oracle-data")

    if args.disjoint_anchor_oracle and anchor_data is None and oracle_data is None:
        rows = _load_plain_rows(data)
        order = _stable_order(rows, salt=f"anchor-oracle:{data}")
        anchor_ids = order[: min(args.anchor_max_rows, len(order))]
        oracle_start = len(anchor_ids)
        oracle_ids = order[oracle_start : oracle_start + args.oracle_max_rows]
        anchor_rows = [rows[i] for i in anchor_ids]
        oracle_rows = [rows[i] for i in oracle_ids]
    elif args.disjoint_anchor_oracle and (
        (anchor_data is None or oracle_data is None)
        or str(anchor_data or data) == str(oracle_data or data)
    ):
        source = anchor_data or oracle_data or data
        rows = _load_plain_rows(source)
        order = _stable_order(rows, salt=f"anchor-oracle:{source}")
        anchor_ids = order[: min(args.anchor_max_rows, len(order))]
        oracle_start = len(anchor_ids)
        oracle_ids = order[oracle_start : oracle_start + args.oracle_max_rows]
        anchor_rows = [rows[i] for i in anchor_ids]
        oracle_rows = [rows[i] for i in oracle_ids]
    else:
        anchor_source = anchor_data or data
        oracle_source = oracle_data or data
        anchor_all = _load_plain_rows(anchor_source)
        oracle_all = _load_plain_rows(oracle_source)
        anchor_ids, anchor_rows = _select_rows(
            anchor_all, max_rows=args.anchor_max_rows, salt=f"anchor:{anchor_source}"
        )
        oracle_ids, oracle_rows = _select_rows(
            oracle_all, max_rows=args.oracle_max_rows, salt=f"oracle:{oracle_source}"
        )

    anchor_hashes = [_row_hash(row) for row in anchor_rows]
    oracle_hashes = [_row_hash(row) for row in oracle_rows]
    overlap = sorted(set(anchor_hashes) & set(oracle_hashes))
    manifest = {
        "anchor_data": str(anchor_data or data),
        "oracle_data": str(oracle_data or data),
        "anchor_row_ids": anchor_ids,
        "oracle_row_ids": oracle_ids,
        "anchor_rows": len(anchor_rows),
        "oracle_rows": len(oracle_rows),
        "overlap_count": len(overlap),
        "overlap_hashes": overlap[:16],
        "anchor_hash": _combined_hash(anchor_rows),
        "oracle_hash": _combined_hash(oracle_rows),
    }
    return anchor_rows, oracle_rows, manifest


def _batches_from_rows(
    rows: list[dict],
    *,
    tokenizer,
    device,
    seq_len: int,
    batch_size: int,
    batches: int,
    train_on: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    ds = build_packed_dataset(
        rows,
        tokenizer,
        learner_id=0,
        num_learners=1,
        seq_len=seq_len,
        max_rows=len(rows),
        train_on=train_on,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    out = []
    for input_ids, weights in loader:
        out.append((input_ids.to(device, non_blocking=True), weights.to(device, non_blocking=True)))
        if len(out) >= batches:
            break
    if not out:
        raise SystemExit("split produced no probe batches")
    return out


def _consensus_update(
    *,
    root: Path,
    group: list[dict],
    frag,
    current_flat: torch.Tensor,
) -> torch.Tensor:
    if not group:
        return torch.zeros_like(current_flat)
    total = 0.0
    merged = torch.zeros_like(current_flat)
    for row in group:
        candidate = _read_f32(_resolve(root, row["candidate_f32"]), frag.numel)
        weight = float(row.get("weight", row.get("c_tokens", 1.0)))
        if weight <= 0.0 or not math.isfinite(weight):
            continue
        merged.add_(candidate, alpha=weight)
        total += weight
    if total <= 0.0:
        return torch.zeros_like(current_flat)
    merged.div_(total)
    return merged - current_flat


def _infer_seed(*paths: Path | None) -> int | None:
    for path in paths:
        if path is None:
            continue
        match = re.search(r"seed(\d+)", str(path))
        if match:
            return int(match.group(1))
    return None


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def annotate(args) -> list[dict]:
    root = args.capture_dir
    rows = _read_jsonl(args.records)
    if args.max_records is not None and len(rows) > args.max_records:
        rows = random.Random(args.sample_seed).sample(rows, args.max_records)
    rows.sort(
        key=lambda r: (
            str(r["capture_state_checkpoint"]),
            int(r.get("pull_step", 0)),
            int(r["fragment"]),
            int(r["learner_id"]),
        )
    )
    capture_groups = _read_capture_index(root)

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
    gradient_cache: dict[tuple[str, int], torch.Tensor] = {}
    consensus_cache: dict[tuple[str, int, int], torch.Tensor] = {}
    anchor_loss = 0.0
    anchor_tokens = 0.0
    out = []

    was_training = model.training
    model.train()
    try:
        for row in rows:
            state_rel = str(row["capture_state_checkpoint"])
            state_path = _resolve(root, state_rel)
            if current_state_path != state_path:
                current_ckpt = parse_checkpoint(state_path)
                syncer_eval._apply_checkpoint(current_ckpt, layout, params, device)
                anchor_loss, anchor_tokens = _anchor_gradient(model, batches, compute_loss)
                gradient_cache.clear()
                consensus_cache.clear()
                current_state_path = state_path
            assert current_ckpt is not None

            fid = int(row["fragment"])
            frag = layout.fragments[fid]
            grad_key = (state_rel, fid)
            if grad_key not in gradient_cache:
                gradient_cache[grad_key] = _fragment_flat_from_grads(frag, params)
            grad = gradient_cache[grad_key]
            current_flat = current_ckpt.fragments[fid][1]
            candidate = _read_f32(_resolve(root, row["capture_candidate_f32"]), frag.numel)
            update = candidate - current_flat
            descent = -grad
            dot = float(torch.dot(descent, update).item()) * args.probe_outer_lr
            grad_norm = float(grad.norm().item())
            update_norm = float(update.norm().item())
            normed_dot = dot / max(grad_norm * update_norm, 1e-12)
            curvature_penalized = dot - args.curvature_lambda * update_norm * update_norm

            group_key = (state_rel, int(row.get("pull_step", 0)), fid)
            if group_key not in consensus_cache:
                consensus_cache[group_key] = _consensus_update(
                    root=root,
                    group=capture_groups.get(group_key, []),
                    frag=frag,
                    current_flat=current_flat,
                )
            consensus = consensus_cache[group_key]
            consensus_norm = float(consensus.norm().item())
            consensus_dot = float(torch.dot(consensus, update).item())
            annotated = dict(row)
            annotated.update(
                {
                    "anchor_loss": anchor_loss,
                    "anchor_tokens": anchor_tokens,
                    "probe_grad_dot": dot,
                    "probe_grad_cosine": _cosine(descent, update),
                    "probe_grad_normed_dot": normed_dot,
                    "curvature_penalized_dot": curvature_penalized,
                    "probe_grad_norm": grad_norm,
                    "consensus_cosine": _cosine(consensus, update),
                    "consensus_normed_dot": consensus_dot / max(consensus_norm * update_norm, 1e-12),
                    "consensus_dot": consensus_dot,
                    "consensus_norm": consensus_norm,
                }
            )
            out.append(annotated)
    finally:
        model.train(was_training)
    return out


def evaluate_disjoint(args) -> tuple[list[dict], dict]:
    root = args.capture_dir
    rows = _read_capture_rows(root)
    if args.max_records is not None and len(rows) > args.max_records:
        rows = random.Random(args.sample_seed).sample(rows, args.max_records)
    rows.sort(
        key=lambda r: (
            str(r["state_checkpoint"]),
            int(r.get("step", 0)),
            int(r["fragment"]),
            int(r["learner_id"]),
        )
    )
    capture_groups = _read_capture_index(root)
    anchor_rows, oracle_rows, manifest = _data_splits(args)
    if args.disjoint_anchor_oracle and manifest["overlap_count"] != 0:
        raise SystemExit(
            "anchor/oracle split overlap_count must be zero; "
            f"got {manifest['overlap_count']}"
        )

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
    anchor_batches = _batches_from_rows(
        anchor_rows,
        tokenizer=tokenizer,
        device=device,
        seq_len=args.seq_len,
        batch_size=args.anchor_batch_size,
        batches=args.anchor_batches,
        train_on=args.train_on,
    )
    oracle_batches = _batches_from_rows(
        oracle_rows,
        tokenizer=tokenizer,
        device=device,
        seq_len=args.seq_len,
        batch_size=args.oracle_batch_size,
        batches=args.oracle_batches,
        train_on=args.train_on,
    )
    manifest["anchor_batches"] = len(anchor_batches)
    manifest["oracle_batches"] = len(oracle_batches)
    compute_loss = lambda logits, ids, w: sft_loss(logits, ids, args.loss_function, w)  # noqa: E731

    current_state_path: Path | None = None
    current_ckpt = None
    gradient_cache: dict[tuple[str, int], torch.Tensor] = {}
    consensus_cache: dict[tuple[str, int, int], torch.Tensor] = {}
    norm_history: dict[int, list[float]] = defaultdict(list)
    norm_ema: dict[int, float] = {}
    norm_var: dict[int, float] = defaultdict(lambda: 1e-4)
    anchor_loss = 0.0
    anchor_tokens = 0.0
    oracle_base_loss = 0.0
    oracle_base_by_batch: list[float] = []
    out = []

    was_training = model.training
    try:
        for row in rows:
            state_rel = str(row["state_checkpoint"])
            state_path = _resolve(root, state_rel)
            if current_state_path != state_path:
                current_ckpt = parse_checkpoint(state_path)
                syncer_eval._apply_checkpoint(current_ckpt, layout, params, device)
                oracle_base_loss, oracle_base_by_batch = syncer_eval._losses(
                    model, oracle_batches, compute_loss
                )
                model.eval()
                anchor_loss, anchor_tokens = _anchor_gradient(model, anchor_batches, compute_loss)
                gradient_cache.clear()
                consensus_cache.clear()
                current_state_path = state_path
            assert current_ckpt is not None

            fid = int(row["fragment"])
            frag = layout.fragments[fid]
            grad_key = (state_rel, fid)
            if grad_key not in gradient_cache:
                gradient_cache[grad_key] = _fragment_flat_from_grads(frag, params)
            grad = gradient_cache[grad_key]
            current_flat = current_ckpt.fragments[fid][1]
            momentum = current_ckpt.fragments[fid][2]
            candidate = _read_f32(_resolve(root, row["candidate_f32"]), frag.numel)
            update = candidate - current_flat
            descent = -grad
            dot = float(torch.dot(descent, update).item()) * args.probe_outer_lr
            grad_norm = float(grad.norm().item())
            update_norm = float(update.norm().item())
            normed_dot = dot / max(grad_norm * update_norm, 1e-12)
            curvature_penalized = dot - args.curvature_lambda * update_norm * update_norm

            hist = norm_history[fid]
            if hist:
                h = torch.tensor(hist, dtype=torch.float32)
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
            if len(hist) > 96:
                del hist[:-96]

            group_key = (state_rel, int(row.get("step", 0)), fid)
            if group_key not in consensus_cache:
                consensus_cache[group_key] = _consensus_update(
                    root=root,
                    group=capture_groups.get(group_key, []),
                    frag=frag,
                    current_flat=current_flat,
                )
            consensus = consensus_cache[group_key]
            consensus_norm = float(consensus.norm().item())
            consensus_dot = float(torch.dot(consensus, update).item())

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
                oracle_batches,
                compute_loss,
                frag,
                params,
                current_flat,
                trial,
                device,
            )
            utility = oracle_base_loss - trial_loss
            utility_se = _utility_se(oracle_base_by_batch, trial_by_batch)

            annotated = {
                "schema": "anchor_gradient_disjoint_v1",
                "seed": args.seed,
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
                "weight": float(row.get("weight", row.get("c_tokens", 1.0))),
                "age": age,
                "freshness": freshness,
                "alignment": alignment,
                "uncertainty": uncertainty,
                "norm_anomaly": norm_anomaly,
                "combined_score": combined_score,
                "update_norm": update_norm,
                "base_loss": oracle_base_loss,
                "trial_loss": trial_loss,
                "utility": utility,
                "utility_se": utility_se,
                "bad": utility < 0.0,
                "bad_strict": None if utility_se is None else utility + utility_se < 0.0,
                "probe_outer_lr": args.probe_outer_lr,
                "capture_state_checkpoint": row["state_checkpoint"],
                "capture_candidate_f32": row["candidate_f32"],
                "anchor_loss": anchor_loss,
                "anchor_tokens": anchor_tokens,
                "anchor_batches": len(anchor_batches),
                "oracle_batches": len(oracle_batches),
                "probe_grad_dot": dot,
                "probe_grad_cosine": _cosine(descent, update),
                "probe_grad_normed_dot": normed_dot,
                "curvature_penalized_dot": curvature_penalized,
                "probe_grad_norm": grad_norm,
                "consensus_cosine": _cosine(consensus, update),
                "consensus_normed_dot": consensus_dot / max(consensus_norm * update_norm, 1e-12),
                "consensus_dot": consensus_dot,
                "consensus_norm": consensus_norm,
            }
            out.append(annotated)
    finally:
        model.train(was_training)
    return out, manifest


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise SystemExit("no anchor-gradient records")
    summarizer = syncer_eval._load_probe_summarizer()
    utility = [float(r["utility"]) for r in rows]
    bad = [u < 0.0 for u in utility]
    good = [not b for b in bad]

    signals = {
        "token_count": [float(r["c_tokens"]) for r in rows],
        "freshness": [float(r["freshness"]) for r in rows],
        "alignment": [float(r["alignment"]) for r in rows],
        "combined_score": [float(r["combined_score"]) for r in rows],
        "probe_grad_dot": [float(r["probe_grad_dot"]) for r in rows],
        "probe_grad_cosine": [float(r["probe_grad_cosine"]) for r in rows],
        "probe_grad_normed_dot": [float(r["probe_grad_normed_dot"]) for r in rows],
        "curvature_penalized_dot": [float(r["curvature_penalized_dot"]) for r in rows],
        "consensus_cosine": [float(r["consensus_cosine"]) for r in rows],
    }
    signal_table = {}
    for name, values in signals.items():
        prob_good = (
            values
            if name == "combined_score"
            else summarizer.calibrated_probability(values)
        )
        signal_table[name] = {
            "pearson_utility": summarizer.pearson(values, utility),
            "spearman_utility": summarizer.spearman(values, utility),
            "bad_fragment_auroc": summarizer.auroc(bad, [-v for v in values]),
            "calibration_error": summarizer.calibration_error(good, prob_good),
        }
    utility_ses = [
        float(r["utility_se"])
        for r in rows
        if r.get("utility_se") is not None and math.isfinite(float(r["utility_se"]))
    ]
    strict = [r.get("bad_strict") for r in rows if r.get("bad_strict") is not None]
    return {
        "records": len(rows),
        "seeds": sorted({int(r["seed"]) for r in rows if r.get("seed") is not None}),
        "negative_utility_rate": sum(bad) / len(rows),
        "bad_strict_rate": sum(1 for v in strict if v) / len(strict) if strict else None,
        "utility_noise_estimate": sum(utility_ses) / len(utility_ses) if utility_ses else None,
        "signals": signal_table,
        "token_auroc": signal_table["token_count"]["bad_fragment_auroc"],
        "hand_score_auroc": signal_table["combined_score"]["bad_fragment_auroc"],
        "probe_grad_dot_auroc": signal_table["probe_grad_dot"]["bad_fragment_auroc"],
        "probe_grad_cosine_auroc": signal_table["probe_grad_cosine"]["bad_fragment_auroc"],
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", nargs="?", type=Path, help="syncer-current candidate utility JSONL")
    p.add_argument("--capture-dir", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--anchor-data", type=Path)
    p.add_argument("--oracle-data", type=Path)
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
    p.add_argument("--anchor-batches", type=int, default=2)
    p.add_argument("--anchor-batch-size", type=int, default=1)
    p.add_argument("--anchor-max-rows", type=int, default=128)
    p.add_argument("--oracle-batches", type=int, default=8)
    p.add_argument("--oracle-batch-size", type=int, default=1)
    p.add_argument("--oracle-max-rows", type=int, default=256)
    p.add_argument("--disjoint-anchor-oracle", action="store_true")
    p.add_argument("--split-manifest-out", type=Path, default=None)
    p.add_argument("--probe-outer-lr", type=float, default=1.0)
    p.add_argument("--probe-freshness-scale", type=float, default=24.0)
    p.add_argument("--curvature-lambda", type=float, default=0.0)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", "--out-jsonl", dest="out", required=True, type=Path)
    p.add_argument("--summary-out", "--out-summary", dest="summary_out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.seed = args.seed if args.seed is not None else _infer_seed(args.capture_dir, args.data, args.out)
    if args.disjoint_anchor_oracle or args.records is None:
        rows, manifest = evaluate_disjoint(args)
        if args.split_manifest_out is not None:
            args.split_manifest_out.parent.mkdir(parents=True, exist_ok=True)
            args.split_manifest_out.write_text(
                json.dumps(_jsonable(manifest), indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
    else:
        if args.data is None:
            raise SystemExit("legacy annotation mode requires --data")
        rows = annotate(args)
        manifest = None

    _write_jsonl(args.out, rows)
    summary = summarize_rows(rows)
    if manifest is not None:
        summary["split_manifest"] = manifest
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(json.dumps(_jsonable({"records": len(rows), "out": str(args.out), "summary": summary}), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
