#!/usr/bin/env python3
"""Replay candidate-probe merge policies on captured syncer groups."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter
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
    "candidate_anchor_positive",
    "candidate_anchor_strict_positive",
    "candidate_anchor_top50",
    "candidate_anchor_drop_strict_bad",
    "candidate_probe_lcb",
    "candidate_probe_v1",
    "candidate_probe_softplus",
    "oracle_positive",
    "oracle_topk",
    "random_positive_count",
    "random_top50_count",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path}: no records")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n")


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_f32(path: Path, expected: int) -> torch.Tensor:
    raw = path.read_bytes()
    if len(raw) != expected * 4:
        raise ValueError(f"{path}: got {len(raw)} bytes, expected {expected * 4}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def infer_seed(*paths: Path | None) -> int | None:
    for path in paths:
        if path is None:
            continue
        match = re.search(r"seed(\d+)", str(path))
        if match:
            return int(match.group(1))
    return None


def jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def std(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def utility_se(base_by_batch: list[float], trial_by_batch: list[float]) -> float | None:
    utilities = [b - t for b, t in zip(base_by_batch, trial_by_batch)]
    if len(utilities) < 2:
        return None
    return std(utilities) / math.sqrt(len(utilities))


def softplus(x: float) -> float:
    if x > 30.0:
        return x
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def group_rows(rows: list[dict], min_candidates: int) -> list[list[dict]]:
    groups: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        key = (str(row["state_checkpoint"]), int(row["step"]), int(row["fragment"]))
        groups.setdefault(key, []).append(row)
    return [
        sorted(group, key=lambda r: int(r["learner_id"]))
        for _, group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0]))
        if len(group) >= min_candidates
    ]


def candidate_key_from_capture(row: dict) -> tuple[str, str, int, int, int]:
    return (
        str(row["state_checkpoint"]),
        str(row["candidate_f32"]),
        int(row["step"]),
        int(row["fragment"]),
        int(row["learner_id"]),
    )


def candidate_key_from_features(row: dict) -> tuple[str, str, int, int, int]:
    return (
        str(row["capture_state_checkpoint"]),
        str(row["capture_candidate_f32"]),
        int(row["pull_step"]),
        int(row["fragment"]),
        int(row["learner_id"]),
    )


def load_candidate_oracle(path: Path | None) -> dict[tuple[str, str, int, int, int], dict]:
    if path is None:
        return {}
    out = {}
    for row in read_jsonl(path):
        out[candidate_key_from_features(row)] = row
    return out


def batches_from_manifest(args, tokenizer, device) -> tuple[list, list, dict]:
    if args.split_manifest is None:
        anchor_rows, oracle_rows, manifest = anchor_eval._data_splits(args)
    else:
        manifest = json.loads(args.split_manifest.read_text())
        source = args.data or Path(manifest["anchor_data"])
        rows = anchor_eval._load_plain_rows(source)
        anchor_ids = [int(i) for i in manifest["anchor_row_ids"]]
        oracle_ids = [int(i) for i in manifest["oracle_row_ids"]]
        anchor_rows = [rows[i] for i in anchor_ids]
        oracle_rows = [rows[i] for i in oracle_ids]
    if args.disjoint_anchor_oracle and int(manifest.get("overlap_count", 0)) != 0:
        raise SystemExit(f"anchor/oracle overlap_count must be 0, got {manifest['overlap_count']}")
    anchor_batches = anchor_eval._batches_from_rows(
        anchor_rows,
        tokenizer=tokenizer,
        device=device,
        seq_len=args.seq_len,
        batch_size=args.anchor_batch_size,
        batches=args.anchor_batches,
        train_on=args.train_on,
    )
    oracle_batches = anchor_eval._batches_from_rows(
        oracle_rows,
        tokenizer=tokenizer,
        device=device,
        seq_len=args.seq_len,
        batch_size=args.oracle_batch_size,
        batches=args.oracle_batches,
        train_on=args.train_on,
    )
    manifest = dict(manifest)
    manifest["anchor_batches"] = len(anchor_batches)
    manifest["oracle_batches"] = len(oracle_batches)
    return anchor_batches, oracle_batches, manifest


def eval_trial(model, batches, compute_loss, frag, params, current_flat, trial_flat, device):
    apply_fragment(frag, trial_flat.to(device), params)
    trial_loss, trial_by_batch = syncer_eval._losses(model, batches, compute_loss)
    apply_fragment(frag, current_flat.to(device), params)
    return trial_loss, trial_by_batch


def token_weight(row: dict) -> float:
    return float(row.get("weight", row.get("c_tokens", 1.0)))


def selected_mass(selected: list[dict], candidates: list[dict]) -> float:
    total = sum(max(token_weight(r), 0.0) for r in candidates)
    if total <= 0.0:
        return 0.0
    return sum(max(token_weight(r), 0.0) for r in selected) / total


def top_fraction(candidates: list[dict], field: str, frac: float) -> list[dict]:
    if not candidates:
        return []
    k = max(1, min(len(candidates), math.ceil(len(candidates) * frac)))
    return sorted(candidates, key=lambda r: float(r[field]), reverse=True)[:k]


def select_candidates(policy: str, candidates: list[dict], args, rng: random.Random) -> tuple[list[dict], str]:
    if policy == "token_weighted":
        return list(candidates), "token"
    if policy == "candidate_anchor_positive":
        return [r for r in candidates if float(r["anchor_utility"]) > args.anchor_threshold], "token"
    if policy == "candidate_anchor_strict_positive":
        return [
            r
            for r in candidates
            if r["anchor_utility_se"] is not None
            and float(r["anchor_utility"]) - args.lcb_z * float(r["anchor_utility_se"])
            > args.anchor_threshold
        ], "token"
    if policy == "candidate_anchor_top50":
        return top_fraction(candidates, "anchor_utility", 0.5), "token"
    if policy == "candidate_anchor_drop_strict_bad":
        return [
            r
            for r in candidates
            if r["anchor_utility_se"] is None
            or float(r["anchor_utility"]) + args.lcb_z * float(r["anchor_utility_se"]) >= 0.0
        ], "token"
    if policy == "candidate_probe_lcb":
        selected = [
            r
            for r in candidates
            if r["anchor_utility_se"] is not None
            and float(r["anchor_utility"]) - args.lcb_z * float(r["anchor_utility_se"])
            > args.anchor_threshold
        ]
        return selected, "token"
    if policy == "candidate_probe_v1":
        selected = [
            r
            for r in candidates
            if r["anchor_utility_se"] is not None
            and float(r["anchor_utility"]) - args.lcb_z * float(r["anchor_utility_se"])
            > args.anchor_threshold
        ]
        if selected_mass(selected, candidates) < args.min_selected_mass:
            selected = top_fraction(candidates, "anchor_utility", 0.5)
        return selected, "token"
    if policy == "candidate_probe_softplus":
        return list(candidates), "anchor_softplus"
    if policy == "oracle_positive":
        return [r for r in candidates if float(r["oracle_individual_utility"]) > 0.0], "token"
    if policy == "oracle_topk":
        return top_fraction(candidates, "oracle_individual_utility", 0.5), "token"
    if policy == "random_positive_count":
        count = sum(1 for r in candidates if float(r["anchor_utility"]) > args.anchor_threshold)
        return rng.sample(candidates, min(len(candidates), count)) if count else [], "token"
    if policy == "random_top50_count":
        count = max(1, min(len(candidates), math.ceil(len(candidates) * 0.5)))
        return rng.sample(candidates, count), "token"
    raise ValueError(f"unknown policy {policy!r}")


def merge_flat(
    *,
    current_flat: torch.Tensor,
    selected: list[dict],
    candidate_tensors: dict[int, torch.Tensor],
    mode: str,
    args,
) -> torch.Tensor:
    if not selected:
        return current_flat.clone()
    total = 0.0
    out = torch.zeros_like(current_flat)
    for row in selected:
        weight = token_weight(row)
        if mode == "anchor_softplus":
            weight *= softplus(float(row["anchor_utility"]) / max(args.tau, 1e-12))
        if weight <= 0.0 or not math.isfinite(weight):
            continue
        out.add_(candidate_tensors[int(row["learner_id"])], alpha=weight)
        total += weight
    if total <= 0.0:
        return current_flat.clone()
    return out.div(total)


def eval_policy(
    *,
    policy: str,
    selected: list[dict],
    mode: str,
    current_flat: torch.Tensor,
    candidate_tensors: dict[int, torch.Tensor],
    model,
    batches,
    compute_loss,
    frag,
    params,
    base_loss: float,
    base_by_batch: list[float],
    device,
    all_candidates: list[dict],
    args,
) -> dict:
    merged = merge_flat(
        current_flat=current_flat,
        selected=selected,
        candidate_tensors=candidate_tensors,
        mode=mode,
        args=args,
    )
    trial = current_flat + args.outer_lr * (merged - current_flat)
    trial_loss, trial_by_batch = eval_trial(
        model, batches, compute_loss, frag, params, current_flat, trial, device
    )
    utility = base_loss - trial_loss
    se = utility_se(base_by_batch, trial_by_batch)
    return {
        f"{policy}_utility": utility,
        f"{policy}_utility_se": se,
        f"{policy}_negative": utility < 0.0,
        f"{policy}_strict_negative": None if se is None else utility + se < 0.0,
        f"{policy}_selected_count": len(selected),
        f"{policy}_selected_mass": selected_mass(selected, all_candidates),
    }


def replay(args) -> tuple[list[dict], dict]:
    root = args.capture_dir
    rows = read_jsonl(root / "index.jsonl")
    groups = group_rows(rows, args.min_candidates)
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
    anchor_batches, oracle_batches, manifest = batches_from_manifest(args, tokenizer, device)
    compute_loss = lambda logits, ids, w: sft_loss(logits, ids, args.loss_function, w)  # noqa: E731
    rng = random.Random(args.sample_seed)
    candidate_oracle = load_candidate_oracle(args.candidate_oracle_features)

    current_state_path: Path | None = None
    current_ckpt = None
    anchor_base_loss = 0.0
    anchor_base_by_batch: list[float] = []
    oracle_base_loss = 0.0
    oracle_base_by_batch: list[float] = []
    records = []
    was_training = model.training
    model.eval()
    try:
        for idx, group in enumerate(groups, start=1):
            first = group[0]
            state_rel = str(first["state_checkpoint"])
            state_path = resolve(root, state_rel)
            if current_state_path != state_path:
                current_ckpt = parse_checkpoint(state_path)
                syncer_eval._apply_checkpoint(current_ckpt, layout, params, device)
                anchor_base_loss, anchor_base_by_batch = syncer_eval._losses(
                    model, anchor_batches, compute_loss
                )
                oracle_base_loss, oracle_base_by_batch = syncer_eval._losses(
                    model, oracle_batches, compute_loss
                )
                current_state_path = state_path
            assert current_ckpt is not None

            fid = int(first["fragment"])
            frag = layout.fragments[fid]
            current_flat = current_ckpt.fragments[fid][1]
            candidate_tensors = {
                int(row["learner_id"]): read_f32(resolve(root, row["candidate_f32"]), frag.numel)
                for row in group
            }
            annotated = []
            for row in group:
                learner_id = int(row["learner_id"])
                candidate = candidate_tensors[learner_id]
                anchor_trial_loss, anchor_trial_by_batch = eval_trial(
                    model,
                    anchor_batches,
                    compute_loss,
                    frag,
                    params,
                    current_flat,
                    current_flat + args.outer_lr * (candidate - current_flat),
                    device,
                )
                item = dict(row)
                item["anchor_utility"] = anchor_base_loss - anchor_trial_loss
                item["anchor_utility_se"] = utility_se(anchor_base_by_batch, anchor_trial_by_batch)
                oracle_row = candidate_oracle.get(candidate_key_from_capture(row))
                if oracle_row is None:
                    oracle_trial_loss, oracle_trial_by_batch = eval_trial(
                        model,
                        oracle_batches,
                        compute_loss,
                        frag,
                        params,
                        current_flat,
                        current_flat + args.outer_lr * (candidate - current_flat),
                        device,
                    )
                    item["oracle_individual_utility"] = oracle_base_loss - oracle_trial_loss
                    item["oracle_individual_utility_se"] = utility_se(
                        oracle_base_by_batch, oracle_trial_by_batch
                    )
                else:
                    item["oracle_individual_utility"] = float(oracle_row["utility"])
                    item["oracle_individual_utility_se"] = oracle_row.get("utility_se")
                annotated.append(item)

            out = {
                "schema": "candidate_probe_policy_replay_v1",
                "seed": args.seed,
                "step": int(first["step"]),
                "fragment": fid,
                "state_checkpoint": state_rel,
                "candidate_count": len(annotated),
                "anchor_base_loss": anchor_base_loss,
                "oracle_base_loss": oracle_base_loss,
                "anchor_utility_mean": mean([float(r["anchor_utility"]) for r in annotated]),
                "oracle_individual_utility_mean": mean(
                    [float(r["oracle_individual_utility"]) for r in annotated]
                ),
                "anchor_bad_rate": mean(
                    [1.0 if float(r["anchor_utility"]) < 0.0 else 0.0 for r in annotated]
                ),
                "oracle_individual_bad_rate": mean(
                    [1.0 if float(r["oracle_individual_utility"]) < 0.0 else 0.0 for r in annotated]
                ),
            }
            for policy in args.policies:
                selected, mode = select_candidates(policy, annotated, args, rng)
                out.update(
                    eval_policy(
                        policy=policy,
                        selected=selected,
                        mode=mode,
                        current_flat=current_flat,
                        candidate_tensors=candidate_tensors,
                        model=model,
                        batches=oracle_batches,
                        compute_loss=compute_loss,
                        frag=frag,
                        params=params,
                        base_loss=oracle_base_loss,
                        base_by_batch=oracle_base_by_batch,
                        device=device,
                        all_candidates=annotated,
                        args=args,
                    )
                )
            token = float(out["token_weighted_utility"])
            oracle_positive = float(out["oracle_positive_utility"])
            oracle_topk = float(out["oracle_topk_utility"])
            out["oracle_positive_headroom"] = oracle_positive - token
            out["oracle_topk_headroom"] = oracle_topk - token
            for policy in args.policies:
                utility = float(out[f"{policy}_utility"])
                denom = oracle_positive - token
                out[f"{policy}_oracle_positive_headroom_captured"] = (
                    None if denom <= 0.0 else (utility - token) / denom
                )
            records.append(out)
            sink = getattr(args, "_record_sink", None)
            if sink is not None:
                sink.write(json.dumps(jsonable(out), sort_keys=True, allow_nan=False) + "\n")
                sink.flush()
            if args.progress_every and (
                idx == 1 or idx % args.progress_every == 0 or idx == len(groups)
            ):
                print(
                    f"[candidate-probe] {idx}/{len(groups)} groups step={out['step']} fragment={fid}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        model.train(was_training)
    return records, manifest


def summarize(records: list[dict]) -> dict:
    if not records:
        raise SystemExit("no records")
    token_neg = mean([1.0 if r["token_weighted_negative"] else 0.0 for r in records])
    token_strict = mean(
        [
            1.0 if r["token_weighted_strict_negative"] else 0.0
            for r in records
            if r["token_weighted_strict_negative"] is not None
        ]
    )
    policies = {}
    policies_in_records = [
        policy for policy in POLICIES if f"{policy}_utility" in records[0]
    ]
    for policy in policies_in_records:
        utilities = [float(r[f"{policy}_utility"]) for r in records]
        gains = [float(r[f"{policy}_utility"]) - float(r["token_weighted_utility"]) for r in records]
        negatives = [1.0 if r[f"{policy}_negative"] else 0.0 for r in records]
        strict = [
            1.0 if r[f"{policy}_strict_negative"] else 0.0
            for r in records
            if r[f"{policy}_strict_negative"] is not None
        ]
        captured = [
            float(v)
            for r in records
            if (v := r.get(f"{policy}_oracle_positive_headroom_captured")) is not None
        ]
        policies[policy] = {
            "groups": len(records),
            "mean_utility": mean(utilities),
            "mean_gain_vs_token": mean(gains),
            "median_gain_vs_token": sorted(gains)[len(gains) // 2],
            "gain_positive_rate": mean([1.0 if g > 0.0 else 0.0 for g in gains]),
            "negative_rate": mean(negatives),
            "negative_rate_relative_drop": (
                None if token_neg <= 0.0 else (token_neg - mean(negatives)) / token_neg
            ),
            "strict_negative_rate": mean(strict),
            "strict_negative_rate_relative_drop": (
                None if token_strict <= 0.0 else (token_strict - mean(strict)) / token_strict
            ),
            "selected_mass_mean": mean([float(r[f"{policy}_selected_mass"]) for r in records]),
            "selected_count_mean": mean([float(r[f"{policy}_selected_count"]) for r in records]),
            "oracle_positive_headroom_captured": mean(captured),
            "headroom_excluded_fraction": 1.0 - len(captured) / len(records),
        }
    main_candidates = [
        p
        for p in policies_in_records
        if not p.startswith("oracle_") and not p.startswith("random_") and p != "token_weighted"
    ]
    main = max(
        main_candidates,
        key=lambda p: policies[p]["mean_gain_vs_token"],
    ) if main_candidates else "token_weighted"
    gates = {
        "main_policy": main,
        "mean_gain_ge_0.0005": policies[main]["mean_gain_vs_token"] >= 0.0005,
        "negative_drop_ge_0.20": policies[main]["negative_rate_relative_drop"] is not None
        and policies[main]["negative_rate_relative_drop"] >= 0.20,
        "strict_drop_ge_0.20": policies[main]["strict_negative_rate_relative_drop"] is not None
        and policies[main]["strict_negative_rate_relative_drop"] >= 0.20,
        "headroom_captured_ge_0.40": policies[main]["oracle_positive_headroom_captured"] >= 0.40,
        "selected_mass_ge_0.40": policies[main]["selected_mass_mean"] >= 0.40,
        "beats_random_positive_count": (
            "random_positive_count" not in policies
            or policies[main]["mean_utility"] > policies["random_positive_count"]["mean_utility"]
        ),
    }
    gates["gate_pass"] = all(v for k, v in gates.items() if k != "main_policy")
    return {
        "schema": "candidate_probe_policy_summary_v1",
        "records": len(records),
        "seeds": sorted({int(r["seed"]) for r in records if r.get("seed") is not None}),
        "candidate_count_mean": mean([float(r["candidate_count"]) for r in records]),
        "anchor_bad_rate_mean": mean([float(r["anchor_bad_rate"]) for r in records]),
        "oracle_individual_bad_rate_mean": mean(
            [float(r["oracle_individual_bad_rate"]) for r in records]
        ),
        "oracle_positive_headroom_mean": mean(
            [float(r["oracle_positive_headroom"]) for r in records]
        ),
        "policies": policies,
        "gates": gates,
    }


def to_markdown(summary: dict) -> str:
    lines = ["# Candidate-Probe Policy Replay", ""]
    lines.append(f"- Records: `{summary['records']}`")
    lines.append(f"- Seeds: `{summary['seeds']}`")
    lines.append(f"- Main policy: `{summary['gates']['main_policy']}`")
    lines.append(f"- Gate pass: `{summary['gates']['gate_pass']}`")
    lines.append("")
    lines.append("| Policy | Gain vs token | Negative drop | Strict drop | Headroom captured | Selected mass |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for policy in summary["policies"]:
        row = summary["policies"][policy]
        lines.append(
            "| `{}` | {:.6f} | {} | {} | {} | {:.3f} |".format(
                policy,
                row["mean_gain_vs_token"],
                fmt(row["negative_rate_relative_drop"]),
                fmt(row["strict_negative_rate_relative_drop"]),
                fmt(row["oracle_positive_headroom_captured"]),
                row["selected_mass_mean"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def fmt(value) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir", required=True, type=Path)
    p.add_argument("--split-manifest", type=Path)
    p.add_argument("--candidate-oracle-features", type=Path)
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
    p.add_argument("--anchor-batches", type=int, default=8)
    p.add_argument("--anchor-batch-size", type=int, default=1)
    p.add_argument("--anchor-max-rows", type=int, default=64)
    p.add_argument("--oracle-batches", type=int, default=8)
    p.add_argument("--oracle-batch-size", type=int, default=1)
    p.add_argument("--oracle-max-rows", type=int, default=256)
    p.add_argument("--disjoint-anchor-oracle", action="store_true")
    p.add_argument("--outer-lr", type=float, default=1.0)
    p.add_argument("--anchor-threshold", type=float, default=0.0)
    p.add_argument("--lcb-z", type=float, default=1.0)
    p.add_argument("--min-selected-mass", type=float, default=0.35)
    p.add_argument("--tau", type=float, default=0.001)
    p.add_argument("--min-candidates", type=int, default=2)
    p.add_argument("--max-groups", type=int)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--seed", type=int)
    p.add_argument("--policies", nargs="+", default=list(POLICIES))
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--out-jsonl", required=True, type=Path)
    p.add_argument("--out-summary", required=True, type=Path)
    p.add_argument("--out-md", type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    missing = sorted(set(args.policies) - set(POLICIES))
    if missing:
        raise SystemExit(f"unknown policies: {missing}")
    for required in ("token_weighted", "oracle_positive"):
        if required not in args.policies:
            args.policies.append(required)
    args.seed = args.seed if args.seed is not None else infer_seed(args.capture_dir, args.split_manifest)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as sink:
        args._record_sink = sink
        records, manifest = replay(args)
    summary = summarize(records)
    summary["split_manifest"] = manifest
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(to_markdown(summary))
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
