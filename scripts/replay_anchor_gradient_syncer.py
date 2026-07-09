#!/usr/bin/env python3
"""Replay fixed anchor-gradient corrections against a disjoint oracle split.

This is deliberately not an action selector. Each policy deterministically
combines the production current-group outer gradient with a fresh gradient of
the current syncer model on a small anchor set, then evaluates that update on
non-overlapping oracle data. The baseline uses the tuned outer LR.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
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


bn = _load_script("replay_buffered_nesterov_syncer")
ag = _load_script("evaluate_anchor_gradient_features")

from yeto.export import parse_checkpoint  # noqa: E402
from yeto.fragments import build_layout  # noqa: E402
from yeto.learner import load_model_and_tokenizer, trainable_params  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402


POLICIES = (
    "token_weighted",
    "anchor_blend05",
    "anchor_blend10",
    "anchor_blend25",
    "anchor_residual10_normmatch",
    "anchor_residual25_normmatch",
    "anchor_pcgrad_normmatch",
    "anchor_conflict_blend10",
    "anchor_only_normmatch",
)


def _mean(values) -> float:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def _quantile(values, p: float) -> float:
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return float("nan")
    return values[max(0, min(len(values) - 1, round(p * (len(values) - 1))))]


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


def _rows(path: Path) -> list[dict]:
    return ag._load_plain_rows(path)


def _data_manifest(anchor_rows: list[dict], oracle_rows: list[dict], args) -> dict:
    anchor_hashes = [ag._row_hash(row) for row in anchor_rows]
    oracle_hashes = [ag._row_hash(row) for row in oracle_rows]
    overlap = sorted(set(anchor_hashes) & set(oracle_hashes))
    return {
        "anchor_data": str(args.anchor_data),
        "oracle_data": str(args.oracle_data),
        "anchor_rows": len(anchor_rows),
        "oracle_rows": len(oracle_rows),
        "overlap_count": len(overlap),
        "overlap_hashes": overlap[:16],
        "anchor_hash": ag._combined_hash(anchor_rows),
        "oracle_hash": ag._combined_hash(oracle_rows),
    }


def _validate_manifest(manifest: dict) -> None:
    overlap_count = int(manifest.get("overlap_count", 0))
    if overlap_count:
        raise SystemExit(f"anchor/oracle content overlap: {overlap_count}")


def _tensor_scaled_anchor(delta: torch.Tensor, gradient: torch.Tensor, frag) -> torch.Tensor:
    out = torch.zeros_like(delta)
    offset = 0
    for numel in bn._tensor_numels(frag):
        end = offset + numel
        delta_slice = delta[offset:end]
        grad_slice = gradient[offset:end]
        grad_norm = float(grad_slice.norm().item())
        delta_norm = float(delta_slice.norm().item())
        if grad_norm >= 1e-12:
            out[offset:end] = grad_slice * (delta_norm / grad_norm)
        offset = end
    return out


def _tensor_normmatch(source: torch.Tensor, target: torch.Tensor, frag) -> torch.Tensor:
    return bn._match_tensor_norms(source, target, frag)


def _pcgrad(delta: torch.Tensor, gradient: torch.Tensor, frag) -> tuple[torch.Tensor, float]:
    out = delta.clone()
    conflicts = 0
    tensors = bn._tensor_numels(frag)
    offset = 0
    for numel in tensors:
        end = offset + numel
        delta_slice = out[offset:end]
        grad_slice = gradient[offset:end]
        dot = float(torch.dot(delta_slice, grad_slice).item())
        grad_sq = float(torch.dot(grad_slice, grad_slice).item())
        if dot < 0.0 and grad_sq >= 1e-12:
            delta_slice.add_(grad_slice, alpha=-dot / grad_sq)
            conflicts += 1
        offset = end
    return _tensor_normmatch(out, delta, frag), conflicts / max(len(tensors), 1)


def _policy_delta(
    policy: str,
    baseline_delta: torch.Tensor,
    anchor_gradient: torch.Tensor,
    frag,
) -> tuple[torch.Tensor, dict]:
    scaled = _tensor_scaled_anchor(baseline_delta, anchor_gradient, frag)
    cosine = bn.soft._cosine(baseline_delta, anchor_gradient)
    info = {
        "selected_count": 1,
        "selected_mass": 1.0,
        "anchor_gradient_cosine": cosine,
        "anchor_gradient_norm": float(anchor_gradient.norm().item()),
        "baseline_delta_norm": float(baseline_delta.norm().item()),
        "conflict_tensor_fraction": 0.0,
    }
    match = re.fullmatch(r"anchor_blend(\d+)", policy)
    if match:
        blend = float(match.group(1)) / 100.0
        return baseline_delta.mul(1.0 - blend).add(scaled, alpha=blend), info
    match = re.fullmatch(r"anchor_residual(\d+)_normmatch", policy)
    if match:
        residual = float(match.group(1)) / 100.0
        mixed = baseline_delta + residual * scaled
        return _tensor_normmatch(mixed, baseline_delta, frag), info
    if policy == "anchor_pcgrad_normmatch":
        corrected, conflict_fraction = _pcgrad(baseline_delta, anchor_gradient, frag)
        info["conflict_tensor_fraction"] = conflict_fraction
        return corrected, info
    if policy == "anchor_conflict_blend10":
        if cosine < 0.0:
            return baseline_delta.mul(0.9).add(scaled, alpha=0.1), info
        return baseline_delta, info
    if policy == "anchor_only_normmatch":
        return scaled, info
    raise ValueError(policy)


def replay(args) -> tuple[list[dict], dict]:
    root = args.capture_dir
    groups = bn.buffered._group_rows(bn.buffered._read_jsonl(root / "index.jsonl"), 1)
    complete = []
    for group in groups:
        learners = [int(row["learner_id"]) for row in group]
        if len(learners) != len(set(learners)):
            raise SystemExit(
                f"duplicate learner in step={group[0]['step']} fragment={group[0]['fragment']}"
            )
        if len(group) == args.expected_candidates:
            complete.append(group)
        elif not args.drop_incomplete_groups:
            raise SystemExit(
                f"incomplete group step={group[0]['step']} fragment={group[0]['fragment']}: "
                f"got {len(group)}, expected {args.expected_candidates}"
            )
    groups = complete[: args.max_groups] if args.max_groups is not None else complete

    anchor_rows = _rows(args.anchor_data)[: args.anchor_max_rows]
    oracle_rows = _rows(args.oracle_data)[: args.oracle_max_rows]
    manifest = _data_manifest(anchor_rows, oracle_rows, args)
    _validate_manifest(manifest)

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
    anchor_batches = ag._batches_from_rows(
        anchor_rows,
        tokenizer=tokenizer,
        device=device,
        seq_len=args.seq_len,
        batch_size=args.anchor_batch_size,
        batches=args.anchor_batches,
        train_on=args.train_on,
    )
    oracle_batches = ag._batches_from_rows(
        oracle_rows,
        tokenizer=tokenizer,
        device=device,
        seq_len=args.seq_len,
        batch_size=args.oracle_batch_size,
        batches=args.oracle_batches,
        train_on=args.train_on,
    )
    compute_loss = lambda logits, ids, weights: sft_loss(logits, ids, args.loss_function, weights)  # noqa: E731

    records = []
    was_training = model.training
    model.eval()
    try:
        for group_idx, group in enumerate(groups, start=1):
            first = group[0]
            checkpoint = parse_checkpoint(bn.buffered._resolve(root, first["state_checkpoint"]))
            ag.syncer_eval._apply_checkpoint(checkpoint, layout, params, device)
            oracle_base, oracle_base_by_batch = ag.syncer_eval._losses(
                model, oracle_batches, compute_loss
            )
            anchor_loss, anchor_tokens = ag._anchor_gradient(
                model, anchor_batches, compute_loss
            )

            fid = int(first["fragment"])
            frag = layout.fragments[fid]
            current = checkpoint.fragments[fid][1]
            momentum = checkpoint.fragments[fid][2]
            current_version = int(checkpoint.fragments[fid][0])
            anchor_gradient = ag._fragment_flat_from_grads(frag, params)
            candidates = []
            for row in group:
                tensor = bn.buffered._read_f32(
                    bn.buffered._resolve(root, row["candidate_f32"]), frag.numel
                )
                candidates.append(
                    bn.buffered._candidate(
                        row, tensor, current, momentum, current_version
                    )
                )
            baseline_update = bn._production_merge_update(candidates, momentum, frag)
            baseline_delta = -baseline_update
            anchor_gradient_norm = float(anchor_gradient.norm().item())
            baseline_delta_norm = float(baseline_delta.norm().item())
            baseline_trial = bn._nesterov_trial(
                current,
                momentum,
                baseline_update,
                args.outer_lr,
                args.outer_momentum,
            )
            baseline_utility, baseline_se = bn._eval(
                model,
                oracle_batches,
                compute_loss,
                frag,
                params,
                current,
                baseline_trial,
                oracle_base,
                oracle_base_by_batch,
                device,
            )
            out = {
                "schema": "anchor_gradient_syncer_replay_v1",
                "seed": args.seed,
                "step": int(first["step"]),
                "fragment": fid,
                "candidate_count": len(candidates),
                "anchor_loss": anchor_loss,
                "anchor_tokens": anchor_tokens,
                "anchor_gradient_norm": anchor_gradient_norm,
                "baseline_delta_norm": baseline_delta_norm,
                "anchor_gradient_norm_ratio": anchor_gradient_norm
                / max(baseline_delta_norm, 1e-12),
                "anchor_gradient_cosine_to_baseline": bn.soft._cosine(
                    anchor_gradient, baseline_delta
                ),
                "oracle_base_loss": oracle_base,
                "token_weighted_utility": baseline_utility,
                "token_weighted_utility_se": baseline_se,
                "token_weighted_negative": baseline_utility < 0.0,
                "token_weighted_strict_negative": (
                    None if baseline_se is None else baseline_utility + baseline_se < 0.0
                ),
                "token_weighted_gain_vs_token": 0.0,
                "token_weighted_selected_mass": 1.0,
            }
            for policy in args.policies:
                delta, info = _policy_delta(
                    policy, baseline_delta, anchor_gradient, frag
                )
                update = -delta
                trial = bn._nesterov_trial(
                    current,
                    momentum,
                    update,
                    args.outer_lr,
                    args.outer_momentum,
                )
                utility, utility_se = bn._eval(
                    model,
                    oracle_batches,
                    compute_loss,
                    frag,
                    params,
                    current,
                    trial,
                    oracle_base,
                    oracle_base_by_batch,
                    device,
                )
                out[f"{policy}_utility"] = utility
                out[f"{policy}_utility_se"] = utility_se
                out[f"{policy}_negative"] = utility < 0.0
                out[f"{policy}_strict_negative"] = (
                    None if utility_se is None else utility + utility_se < 0.0
                )
                out[f"{policy}_gain_vs_token"] = utility - baseline_utility
                out[f"{policy}_selected_mass"] = 1.0
                out[f"{policy}_delta_norm_ratio"] = float(delta.norm().item()) / max(
                    float(baseline_delta.norm().item()), 1e-12
                )
                out[f"{policy}_delta_cosine_to_baseline"] = bn.soft._cosine(
                    delta, baseline_delta
                )
                for key, value in info.items():
                    out[f"{policy}_{key}"] = value
            records.append(out)
            args._sink.write(json.dumps(_jsonable(out), sort_keys=True) + "\n")
            args._sink.flush()
            if args.progress_every and (len(records) == 1 or len(records) % args.progress_every == 0):
                print(
                    f"[anchor-gradient] records={len(records)} groups={group_idx}/{len(groups)} "
                    f"step={out['step']} fragment={fid}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        model.train(was_training)
    return records, manifest


def summarize(records: list[dict], policies: tuple[str, ...]) -> dict:
    baseline_neg = _mean(row["token_weighted_negative"] for row in records)
    baseline_strict = _mean(
        row["token_weighted_strict_negative"]
        for row in records
        if row["token_weighted_strict_negative"] is not None
    )
    results = {}
    for policy in policies:
        gains = [float(row[f"{policy}_gain_vs_token"]) for row in records]
        negative = _mean(row[f"{policy}_negative"] for row in records)
        strict = _mean(
            row[f"{policy}_strict_negative"]
            for row in records
            if row[f"{policy}_strict_negative"] is not None
        )
        results[policy] = {
            "mean_utility": _mean(row[f"{policy}_utility"] for row in records),
            "mean_gain_vs_token": _mean(gains),
            "median_gain_vs_token": _quantile(gains, 0.5),
            "gain_positive_rate": _mean(gain > 0.0 for gain in gains),
            "negative_rate": negative,
            "negative_rate_relative_drop": (
                None if baseline_neg <= 0.0 else (baseline_neg - negative) / baseline_neg
            ),
            "strict_negative_rate": strict,
            "strict_negative_rate_relative_drop": (
                None
                if baseline_strict <= 0.0
                else (baseline_strict - strict) / baseline_strict
            ),
            "delta_norm_ratio": _mean(
                row.get(f"{policy}_delta_norm_ratio", 1.0) for row in records
            ),
            "delta_cosine_to_baseline": _mean(
                row.get(f"{policy}_delta_cosine_to_baseline", 1.0) for row in records
            ),
            "anchor_gradient_cosine": _mean(
                row.get(f"{policy}_anchor_gradient_cosine", 0.0) for row in records
            ),
            "conflict_tensor_fraction": _mean(
                row.get(f"{policy}_conflict_tensor_fraction", 0.0)
                for row in records
            ),
        }
    non_token = [policy for policy in policies if policy != "token_weighted"]
    best = max(non_token, key=lambda policy: results[policy]["mean_gain_vs_token"])
    return {
        "schema": "anchor_gradient_syncer_summary_v1",
        "records": len(records),
        "seeds": sorted({int(row["seed"]) for row in records}),
        "baseline_negative_rate": baseline_neg,
        "baseline_strict_negative_rate": baseline_strict,
        "anchor_gradient_norm_ratio": _mean(
            row.get("anchor_gradient_norm_ratio") for row in records
        ),
        "anchor_gradient_cosine_to_baseline": _mean(
            row.get("anchor_gradient_cosine_to_baseline") for row in records
        ),
        "policies": results,
        "best_non_token_policy": best,
        "gate": {
            "best_mean_gain_positive": results[best]["mean_gain_vs_token"] > 0.0,
            "best_negative_drop_nonnegative": (
                results[best]["negative_rate_relative_drop"] or 0.0
            ) >= 0.0,
            "best_strict_drop_nonnegative": (
                results[best]["strict_negative_rate_relative_drop"] or 0.0
            ) >= 0.0,
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--anchor-data", required=True, type=Path)
    parser.add_argument("--oracle-data", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tuning", choices=["lora", "full"], default="lora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-targets", choices=["auto", "attention", "all-linear"], default="auto")
    parser.add_argument("--fragments", type=int, default=4)
    parser.add_argument("--fragment-pattern", choices=["binpack", "strided"], default="binpack")
    parser.add_argument("--loss-function", default="cross_entropy")
    parser.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    parser.add_argument("--anchor-batches", type=int, default=2)
    parser.add_argument("--anchor-batch-size", type=int, default=1)
    parser.add_argument("--anchor-max-rows", type=int, default=64)
    parser.add_argument("--oracle-batches", type=int, default=8)
    parser.add_argument("--oracle-batch-size", type=int, default=1)
    parser.add_argument("--oracle-max-rows", type=int, default=256)
    parser.add_argument("--outer-lr", type=float, default=0.35)
    parser.add_argument("--outer-momentum", type=float, default=0.9)
    parser.add_argument("--expected-candidates", type=int, default=4)
    parser.add_argument("--drop-incomplete-groups", action="store_true")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--policies",
        default=",".join(POLICIES[1:]),
        help="Comma-separated non-baseline policies to evaluate.",
    )
    parser.add_argument("--split-manifest-out", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    args = parser.parse_args(argv)
    requested = tuple(dict.fromkeys(part.strip() for part in args.policies.split(",") if part.strip()))
    unknown = sorted(set(requested) - set(POLICIES[1:]))
    if unknown:
        parser.error(f"unknown anchor-gradient policies: {','.join(unknown)}")
    if not requested:
        parser.error("--policies must contain at least one policy")
    args.policies = requested
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    args.seed = args.seed if args.seed is not None else _infer_seed(
        args.capture_dir, args.out_jsonl
    )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as sink:
        args._sink = sink
        records, manifest = replay(args)
    summary = summarize(records, ("token_weighted", *args.policies))
    summary["outer_lr"] = args.outer_lr
    summary["outer_momentum"] = args.outer_momentum
    summary["split_manifest"] = manifest
    args.out_summary.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    args.split_manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.split_manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
