#!/usr/bin/env python3
"""Replay anchor-gradient merge policies on captured syncer groups."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
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
    spec.loader.exec_module(mod)
    return mod


anchor_eval = _load_script("evaluate_anchor_gradient_features")
syncer_eval = anchor_eval.syncer_eval

from yeto.export import parse_checkpoint  # noqa: E402
from yeto.fragments import build_layout  # noqa: E402
from yeto.learner import load_model_and_tokenizer, trainable_params  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402
from yeto.tensor_io import apply_fragment  # noqa: E402


POLICIES = (
    "token_weighted",
    "freshness_weighted",
    "metadata_calibrated",
    "anchor_reweight_sigmoid",
    "anchor_reweight_softplus",
    "anchor_top50",
    "anchor_drop_bottom25",
    "anchor_drop_bottom50",
    "anchor_positive_threshold",
    "anchor_shrink",
    "probecommit_v1",
    "oracle_positive",
    "oracle_topk",
    "random_probecommit_count",
    "random_oracle_positive_count",
)


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


def _utility_se(base_by_batch: list[float], trial_by_batch: list[float]) -> float | None:
    utilities = [b - t for b, t in zip(base_by_batch, trial_by_batch)]
    if len(utilities) < 2:
        return None
    mean = sum(utilities) / len(utilities)
    var = sum((u - mean) ** 2 for u in utilities) / (len(utilities) - 1)
    return math.sqrt(var / len(utilities))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softplus(x: float) -> float:
    if x > 30.0:
        return x
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _quantile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return ordered[idx]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _group_rows(rows: list[dict], *, min_candidates: int) -> list[list[dict]]:
    groups: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        key = (
            str(row["capture_state_checkpoint"]),
            int(row["pull_step"]),
            int(row["fragment"]),
        )
        groups.setdefault(key, []).append(row)
    out = [
        sorted(group, key=lambda r: int(r["learner_id"]))
        for _, group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0]))
        if len(group) >= min_candidates
    ]
    return out


def _oracle_rows_from_manifest(args) -> tuple[list[dict], dict | None]:
    if args.split_manifest is None:
        _, oracle_rows, manifest = anchor_eval._data_splits(args)
        return oracle_rows, manifest
    manifest = json.loads(args.split_manifest.read_text())
    source = args.oracle_data or Path(manifest["oracle_data"])
    all_rows = anchor_eval._load_plain_rows(source)
    oracle_ids = [int(i) for i in manifest["oracle_row_ids"]]
    oracle_rows = [all_rows[i] for i in oracle_ids]
    return oracle_rows, manifest


def _score(row: dict, name: str) -> float:
    return float(row.get(name, 0.0))


def _token_weight(row: dict) -> float:
    return float(row.get("weight", row.get("c_tokens", 1.0)))


def _selected_mass(selected: list[dict], candidates: list[dict]) -> float:
    total = sum(max(_token_weight(r), 0.0) for r in candidates)
    if total <= 0.0:
        return 0.0
    return sum(max(_token_weight(r), 0.0) for r in selected) / total


def _top_fraction(candidates: list[dict], score_field: str, frac: float) -> list[dict]:
    if not candidates:
        return []
    k = max(1, min(len(candidates), math.ceil(len(candidates) * frac)))
    return sorted(candidates, key=lambda r: _score(r, score_field), reverse=True)[:k]


def _policy_candidates(
    *,
    policy: str,
    candidates: list[dict],
    score_field: str,
    percentile: float,
    threshold: float,
    min_selected_mass: float,
    rng: random.Random,
) -> tuple[list[dict], float]:
    outer_lr = 1.0
    if policy in {
        "token_weighted",
        "freshness_weighted",
        "metadata_calibrated",
        "anchor_reweight_sigmoid",
        "anchor_reweight_softplus",
    }:
        return list(candidates), outer_lr
    if policy == "anchor_top50":
        return _top_fraction(candidates, score_field, 0.5), outer_lr
    if policy == "anchor_drop_bottom25":
        return _top_fraction(candidates, score_field, 0.75), outer_lr
    if policy == "anchor_drop_bottom50":
        return _top_fraction(candidates, score_field, 0.5), outer_lr
    if policy == "anchor_positive_threshold":
        selected = [r for r in candidates if _score(r, score_field) >= threshold]
        return selected, outer_lr
    if policy == "anchor_shrink":
        mean_score = _mean([_score(r, score_field) for r in candidates])
        return list(candidates), (0.5 if mean_score < threshold else 1.0)
    if policy == "probecommit_v1":
        scores = [_score(r, score_field) for r in candidates]
        cutoff = _quantile(scores, percentile)
        selected = [r for r in candidates if _score(r, score_field) >= cutoff]
        if _selected_mass(selected, candidates) < min_selected_mass:
            selected = _top_fraction(candidates, score_field, 0.5)
        mean_score = _mean([_score(r, score_field) for r in selected])
        return selected, (0.5 if mean_score < threshold else 1.0)
    if policy == "oracle_positive":
        return [r for r in candidates if float(r["utility"]) > 0.0], outer_lr
    if policy == "oracle_topk":
        return sorted(candidates, key=lambda r: float(r["utility"]), reverse=True)[
            : max(1, math.ceil(len(candidates) * 0.5))
        ], outer_lr
    if policy == "random_probecommit_count":
        selected, _ = _policy_candidates(
            policy="probecommit_v1",
            candidates=candidates,
            score_field=score_field,
            percentile=percentile,
            threshold=threshold,
            min_selected_mass=min_selected_mass,
            rng=rng,
        )
        return rng.sample(candidates, len(selected)) if selected else [], outer_lr
    if policy == "random_oracle_positive_count":
        count = sum(1 for r in candidates if float(r["utility"]) > 0.0)
        return rng.sample(candidates, min(len(candidates), count)) if count else [], outer_lr
    raise ValueError(f"unknown policy {policy!r}")


def _candidate_weight(row: dict, policy: str, *, score_field: str, tau: float, threshold: float) -> float:
    base = _token_weight(row)
    if policy == "freshness_weighted":
        return base * max(float(row.get("freshness", 0.0)), 0.0)
    if policy == "metadata_calibrated":
        return base * max(float(row.get("calibrated_score", row.get("combined_score", 0.0))), 0.0)
    if policy == "anchor_reweight_sigmoid":
        return base * _sigmoid((_score(row, score_field) - threshold) / max(tau, 1e-12))
    if policy in {"anchor_reweight_softplus", "probecommit_v1"}:
        return base * _softplus(_score(row, score_field) / max(tau, 1e-12))
    return base


def _merged_flat(
    *,
    current_flat: torch.Tensor,
    candidate_rows: list[dict],
    candidate_tensors: dict[tuple[int, int], torch.Tensor],
    policy: str,
    score_field: str,
    tau: float,
    threshold: float,
) -> torch.Tensor:
    if not candidate_rows:
        return current_flat.clone()
    out = torch.zeros_like(current_flat)
    total = 0.0
    for row in candidate_rows:
        weight = _candidate_weight(row, policy, score_field=score_field, tau=tau, threshold=threshold)
        if weight <= 0.0 or not math.isfinite(weight):
            continue
        key = (int(row["learner_id"]), int(row["fragment"]))
        out.add_(candidate_tensors[key], alpha=weight)
        total += weight
    if total <= 0.0:
        return current_flat.clone()
    return out.div(total)


def _eval_policy(
    *,
    name: str,
    selected: list[dict],
    current_flat: torch.Tensor,
    candidate_tensors: dict[tuple[int, int], torch.Tensor],
    policy_for_weights: str,
    score_field: str,
    tau: float,
    threshold: float,
    outer_lr: float,
    model,
    batches,
    compute_loss,
    frag,
    params,
    base_loss: float,
    base_by_batch: list[float],
    device,
    all_candidates: list[dict],
) -> dict:
    merged = _merged_flat(
        current_flat=current_flat,
        candidate_rows=selected,
        candidate_tensors=candidate_tensors,
        policy=policy_for_weights,
        score_field=score_field,
        tau=tau,
        threshold=threshold,
    )
    trial = current_flat + outer_lr * (merged - current_flat)
    apply_fragment(frag, trial.to(device), params)
    trial_loss, trial_by_batch = syncer_eval._losses(model, batches, compute_loss)
    apply_fragment(frag, current_flat.to(device), params)
    utility = base_loss - trial_loss
    utility_se = _utility_se(base_by_batch, trial_by_batch)
    return {
        f"{name}_utility": utility,
        f"{name}_utility_se": utility_se,
        f"{name}_negative": utility < 0.0,
        f"{name}_strict_negative": None if utility_se is None else utility + utility_se < 0.0,
        f"{name}_selected_count": len(selected),
        f"{name}_selected_mass": _selected_mass(selected, all_candidates),
        f"{name}_outer_lr_multiplier": outer_lr,
    }


def replay(args) -> list[dict]:
    rows = _read_jsonl(args.features)
    groups = _group_rows(rows, min_candidates=args.min_candidates)
    if args.max_groups is not None and len(groups) > args.max_groups:
        rng = random.Random(args.sample_seed)
        groups = rng.sample(groups, args.max_groups)
        groups.sort(key=lambda g: (int(g[0]["pull_step"]), int(g[0]["fragment"])))

    oracle_rows, manifest = _oracle_rows_from_manifest(args)
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
    batches = anchor_eval._batches_from_rows(
        oracle_rows,
        tokenizer=tokenizer,
        device=device,
        seq_len=args.seq_len,
        batch_size=args.oracle_batch_size,
        batches=args.oracle_batches,
        train_on=args.train_on,
    )
    compute_loss = lambda logits, ids, w: sft_loss(logits, ids, args.loss_function, w)  # noqa: E731
    rng = random.Random(args.sample_seed)

    current_state_path: Path | None = None
    current_ckpt = None
    base_loss = 0.0
    base_by_batch: list[float] = []
    records = []
    was_training = model.training
    try:
        for idx, group in enumerate(groups, start=1):
            first = group[0]
            state_rel = str(first["capture_state_checkpoint"])
            state_path = _resolve(args.capture_dir, state_rel)
            if current_state_path != state_path:
                current_ckpt = parse_checkpoint(state_path)
                syncer_eval._apply_checkpoint(current_ckpt, layout, params, device)
                base_loss, base_by_batch = syncer_eval._losses(model, batches, compute_loss)
                current_state_path = state_path
            assert current_ckpt is not None

            fid = int(first["fragment"])
            frag = layout.fragments[fid]
            current_flat = current_ckpt.fragments[fid][1]
            candidate_tensors = {
                (int(row["learner_id"]), fid): _read_f32(
                    _resolve(args.capture_dir, row["capture_candidate_f32"]), frag.numel
                )
                for row in group
            }
            out = {
                "schema": "probecommit_policy_replay_v1",
                "seed": args.seed,
                "step": int(first["pull_step"]),
                "fragment": fid,
                "state_checkpoint": state_rel,
                "candidate_count": len(group),
                "base_loss": base_loss,
                "score_field": args.score_field,
            }
            for policy in POLICIES:
                selected, outer_lr = _policy_candidates(
                    policy=policy,
                    candidates=group,
                    score_field=args.score_field,
                    percentile=args.drop_percentile,
                    threshold=args.threshold,
                    min_selected_mass=args.min_selected_mass,
                    rng=rng,
                )
                weight_policy = policy
                if policy.startswith("oracle_") or policy.startswith("random_") or policy.startswith("anchor_drop") or policy in {
                    "anchor_top50",
                    "anchor_positive_threshold",
                    "anchor_shrink",
                }:
                    weight_policy = "token_weighted"
                out.update(
                    _eval_policy(
                        name=policy,
                        selected=selected,
                        current_flat=current_flat,
                        candidate_tensors=candidate_tensors,
                        policy_for_weights=weight_policy,
                        score_field=args.score_field,
                        tau=args.tau,
                        threshold=args.threshold,
                        outer_lr=outer_lr,
                        model=model,
                        batches=batches,
                        compute_loss=compute_loss,
                        frag=frag,
                        params=params,
                        base_loss=base_loss,
                        base_by_batch=base_by_batch,
                        device=device,
                        all_candidates=group,
                    )
                )
            token = float(out["token_weighted_utility"])
            oracle_positive = float(out["oracle_positive_utility"])
            oracle_topk = float(out["oracle_topk_utility"])
            out["oracle_positive_headroom"] = oracle_positive - token
            out["oracle_topk_headroom"] = oracle_topk - token
            for policy in POLICIES:
                policy_utility = float(out[f"{policy}_utility"])
                denom_pos = oracle_positive - token
                denom_topk = oracle_topk - token
                out[f"{policy}_oracle_positive_headroom_captured"] = (
                    None if denom_pos <= 0.0 else (policy_utility - token) / denom_pos
                )
                out[f"{policy}_oracle_topk_headroom_captured"] = (
                    None if denom_topk <= 0.0 else (policy_utility - token) / denom_topk
                )
            records.append(out)
            sink = getattr(args, "_record_sink", None)
            if sink is not None:
                sink.write(json.dumps(_jsonable(out), sort_keys=True, allow_nan=False) + "\n")
                sink.flush()
            if args.progress_every and (
                idx == 1 or idx % args.progress_every == 0 or idx == len(groups)
            ):
                print(
                    f"[policy] {idx}/{len(groups)} groups step={out['step']} fragment={fid}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        model.train(was_training)
    if manifest is not None:
        args._manifest = manifest
    return records


def summarize(records: list[dict]) -> dict:
    if not records:
        raise SystemExit("no policy replay records")
    policies = {}
    token_neg = _mean([1.0 if r["token_weighted_negative"] else 0.0 for r in records])
    token_strict = _mean(
        [
            1.0 if r["token_weighted_strict_negative"] else 0.0
            for r in records
            if r["token_weighted_strict_negative"] is not None
        ]
    )
    for policy in POLICIES:
        utilities = [float(r[f"{policy}_utility"]) for r in records]
        negatives = [1.0 if r[f"{policy}_negative"] else 0.0 for r in records]
        strict = [
            1.0 if r[f"{policy}_strict_negative"] else 0.0
            for r in records
            if r[f"{policy}_strict_negative"] is not None
        ]
        captured_pos = [
            float(v)
            for r in records
            if (v := r.get(f"{policy}_oracle_positive_headroom_captured")) is not None
        ]
        captured_topk = [
            float(v)
            for r in records
            if (v := r.get(f"{policy}_oracle_topk_headroom_captured")) is not None
        ]
        policies[policy] = {
            "mean_utility": _mean(utilities),
            "utility_std": _std(utilities),
            "negative_merge_rate": _mean(negatives),
            "strict_negative_merge_rate": _mean(strict),
            "negative_rate_relative_drop": (
                None if token_neg <= 0.0 else (token_neg - _mean(negatives)) / token_neg
            ),
            "strict_negative_rate_relative_drop": (
                None if token_strict <= 0.0 else (token_strict - _mean(strict)) / token_strict
            ),
            "oracle_positive_headroom_captured": _mean(captured_pos),
            "oracle_topk_headroom_captured": _mean(captured_topk),
            "headroom_excluded_fraction_positive": 1.0 - len(captured_pos) / len(records),
            "headroom_excluded_fraction_topk": 1.0 - len(captured_topk) / len(records),
            "selected_mass_mean": _mean([float(r[f"{policy}_selected_mass"]) for r in records]),
            "selected_count_mean": _mean([float(r[f"{policy}_selected_count"]) for r in records]),
        }
    seed_values = sorted({int(r["seed"]) for r in records if r.get("seed") is not None})
    gates = {
        "records": len(records),
        "probecommit_headroom_positive_ge_50pct": policies["probecommit_v1"][
            "oracle_positive_headroom_captured"
        ]
        >= 0.50,
        "probecommit_negative_drop_ge_25pct": policies["probecommit_v1"][
            "negative_rate_relative_drop"
        ]
        is not None
        and policies["probecommit_v1"]["negative_rate_relative_drop"] >= 0.25,
        "probecommit_strict_negative_decreases": policies["probecommit_v1"][
            "strict_negative_merge_rate"
        ]
        < policies["token_weighted"]["strict_negative_merge_rate"],
        "probecommit_selected_mass_ge_40pct": policies["probecommit_v1"][
            "selected_mass_mean"
        ]
        >= 0.40,
        "probecommit_beats_random_count": policies["probecommit_v1"]["mean_utility"]
        > policies["random_probecommit_count"]["mean_utility"],
    }
    gates["gate_c_single_seed_pass"] = all(
        bool(value) for key, value in gates.items() if key != "records"
    )
    return {
        "records": len(records),
        "seeds": seed_values,
        "candidate_count_mean": _mean([float(r["candidate_count"]) for r in records]),
        "oracle_positive_headroom_mean": _mean(
            [float(r["oracle_positive_headroom"]) for r in records]
        ),
        "oracle_topk_headroom_mean": _mean([float(r["oracle_topk_headroom"]) for r in records]),
        "policies": policies,
        "gates": gates,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--capture-dir", required=True, type=Path)
    p.add_argument("--split-manifest", type=Path, default=None)
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
    p.add_argument("--oracle-batches", type=int, default=8)
    p.add_argument("--oracle-batch-size", type=int, default=1)
    p.add_argument("--oracle-max-rows", type=int, default=256)
    p.add_argument("--anchor-max-rows", type=int, default=128)
    p.add_argument("--disjoint-anchor-oracle", action="store_true")
    p.add_argument("--score-field", default="probe_grad_dot")
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--drop-percentile", type=float, default=0.30)
    p.add_argument("--min-selected-mass", type=float, default=0.35)
    p.add_argument("--min-candidates", type=int, default=2)
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--out-jsonl", "--out", dest="out_jsonl", required=True, type=Path)
    p.add_argument("--out-summary", "--summary-out", dest="out_summary", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.seed = args.seed if args.seed is not None else _infer_seed(args.capture_dir, args.features, args.out_jsonl)
    existing = _read_jsonl(args.out_jsonl) if args.resume and args.out_jsonl.exists() else []
    if existing:
        done = {
            (str(r["state_checkpoint"]), int(r["step"]), int(r["fragment"]))
            for r in existing
        }
        all_rows = _read_jsonl(args.features)
        kept = [
            r
            for r in all_rows
            if (
                str(r["capture_state_checkpoint"]),
                int(r["pull_step"]),
                int(r["fragment"]),
            )
            not in done
        ]
        tmp = args.out_jsonl.with_suffix(".remaining_features.jsonl")
        with tmp.open("w") as f:
            for row in kept:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        args.features = tmp

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and existing else "w"
    with args.out_jsonl.open(mode) as f:
        args._record_sink = f
        records = replay(args)
    all_records = existing + records
    summary = summarize(all_records)
    if hasattr(args, "_manifest"):
        summary["split_manifest"] = args._manifest
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
