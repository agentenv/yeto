#!/usr/bin/env python3
"""Add syncer-current anchor-gradient features to captured candidate records."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path, help="syncer-current candidate utility JSONL")
    p.add_argument("--capture-dir", required=True, type=Path)
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
    p.add_argument("--probe-outer-lr", type=float, default=1.0)
    p.add_argument("--curvature-lambda", type=float, default=0.0)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = annotate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"records": len(rows), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
