#!/usr/bin/env python3
"""Replay direct action-probe policies on captured syncer groups.

For each complete `(step, fragment)` group, this script evaluates deployable
merge actions on an anchor split, chooses actions from anchor utility, and
reports the chosen action on a disjoint oracle split. Oracle actions are kept
only as references and are never selectable by action-probe policies.
"""

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


rpp = _load_script("replay_probecommit_policy")
anchor_eval = rpp.anchor_eval
syncer_eval = rpp.syncer_eval

from yeto.export import parse_checkpoint  # noqa: E402
from yeto.fragments import build_layout  # noqa: E402
from yeto.learner import load_model_and_tokenizer, trainable_params  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402


DEFAULT_DEPLOYABLE_ACTIONS = (
    "token_weighted",
    "freshness_weighted",
    "anchor_drop_bottom25",
    "anchor_positive_threshold",
    "anchor_shrink",
    "probecommit_v1",
)

EXTRA_DEPLOYABLE_ACTIONS = (
    "anchor_drop_bottom50",
    "anchor_top50",
    "anchor_reweight_softplus",
    "anchor_reweight_sigmoid",
)

REFERENCE_ACTIONS = (
    "oracle_positive",
    "oracle_topk",
    "random_probecommit_count",
    "random_oracle_positive_count",
)

ACTION_PROBE_POLICIES = (
    "action_probe_top1",
    "action_probe_margin_gated",
    "action_probe_risk_aware",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n")


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
        return 0.0 if vals else float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def infer_seed(*paths: Path | None, row: dict | None = None) -> int | None:
    if row is not None and row.get("seed") is not None:
        return int(row["seed"])
    for path in paths:
        if path is None:
            continue
        match = re.search(r"seed(\d+)", str(path))
        if match:
            return int(match.group(1))
    return None


def replay_key(row: dict) -> tuple[int, int]:
    return int(row["step"]), int(row["fragment"])


def candidate_key(row: dict) -> tuple[int, int]:
    return int(row.get("pull_step", row.get("syncer_global_step", 0))), int(row["fragment"])


def first_jsonl_row(path: Path) -> dict:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise SystemExit(f"{path}: no rows")


def resolve_candidate_features(args) -> Path:
    if args.candidate_features is not None:
        return args.candidate_features
    first = first_jsonl_row(args.features)
    if "capture_state_checkpoint" in first and "capture_candidate_f32" in first:
        return args.features
    if args.policy_replay is not None:
        sibling = args.policy_replay.parent / "anchor_gradient_disjoint.jsonl"
        if sibling.exists():
            return sibling
    raise SystemExit(
        "--features does not look like per-candidate capture features; "
        "pass --candidate-features explicitly"
    )


def validate_manifest(manifest: dict, *, require_disjoint: bool) -> None:
    if require_disjoint and int(manifest.get("overlap_count", -1)) != 0:
        raise SystemExit(
            "anchor/oracle split overlap_count must be zero; "
            f"got {manifest.get('overlap_count')}"
        )


def load_oracle_replay(path: Path | None) -> dict[tuple[int, int], dict]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    return {replay_key(row): row for row in rows}


def data_splits(args) -> tuple[list[dict], list[dict], dict]:
    if args.split_manifest is None:
        return anchor_eval._data_splits(args)
    manifest = json.loads(args.split_manifest.read_text())
    anchor_source = args.anchor_data or args.data or Path(manifest["anchor_data"])
    oracle_source = args.oracle_data or args.data or Path(manifest["oracle_data"])
    anchor_all = anchor_eval._load_plain_rows(anchor_source)
    oracle_all = anchor_eval._load_plain_rows(oracle_source)
    anchor_ids = [int(i) for i in manifest["anchor_row_ids"]]
    oracle_ids = [int(i) for i in manifest["oracle_row_ids"]]
    out = dict(manifest)
    out["anchor_data_local"] = str(anchor_source)
    out["oracle_data_local"] = str(oracle_source)
    return (
        [anchor_all[i] for i in anchor_ids],
        [oracle_all[i] for i in oracle_ids],
        out,
    )


def selected_action_by_anchor(record: dict, actions: tuple[str, ...]) -> tuple[str, float, float]:
    ranked = sorted(
        ((float(record[f"{action}_anchor_utility"]), action) for action in actions),
        reverse=True,
    )
    best_utility, best_action = ranked[0]
    second_utility = ranked[1][0] if len(ranked) > 1 else -float("inf")
    return best_action, best_utility, best_utility - second_utility


def margin_gated_choice(record: dict, actions: tuple[str, ...], margin: float) -> str:
    best, _, gap = selected_action_by_anchor(record, actions)
    return best if gap >= margin else "token_weighted"


def risk_aware_choice(record: dict, actions: tuple[str, ...], threshold: float, fallback: str) -> str:
    best, utility, _ = selected_action_by_anchor(record, actions)
    return fallback if utility < threshold else best


def action_metric(record: dict, action: str, scope: str) -> dict:
    prefix = f"{action}_{scope}"
    return {
        "utility": float(record[f"{prefix}_utility"]),
        "utility_se": record.get(f"{prefix}_utility_se"),
        "negative": bool(record[f"{prefix}_negative"]),
        "strict_negative": record.get(f"{prefix}_strict_negative"),
        "selected_mass": float(record.get(f"{action}_selected_mass", 1.0)),
        "selected_count": float(record.get(f"{action}_selected_count", record["candidate_count"])),
    }


def _weight_policy(policy: str) -> str:
    if (
        policy.startswith("oracle_")
        or policy.startswith("random_")
        or policy.startswith("anchor_drop")
        or policy in {"anchor_top50", "anchor_positive_threshold", "anchor_shrink"}
    ):
        return "token_weighted"
    return policy


def _copy_oracle_metrics(record: dict, replay: dict, actions: tuple[str, ...]) -> None:
    for action in actions:
        for suffix in (
            "utility",
            "utility_se",
            "negative",
            "strict_negative",
            "selected_count",
            "selected_mass",
            "outer_lr_multiplier",
        ):
            key = f"{action}_{suffix}"
            if key in replay:
                record[f"{action}_oracle_{suffix}"] = replay[key]
                if suffix in {"selected_count", "selected_mass", "outer_lr_multiplier"}:
                    record.setdefault(f"{action}_{suffix}", replay[key])


def _eval_action_scope(
    *,
    action: str,
    scope: str,
    selected: list[dict],
    outer_lr: float,
    current_flat,
    candidate_tensors,
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
    weight_policy = _weight_policy(action)
    merged = rpp._merged_flat(
        current_flat=current_flat,
        candidate_rows=selected,
        candidate_tensors=candidate_tensors,
        policy=weight_policy,
        score_field=args.score_field,
        tau=args.tau,
        threshold=args.threshold,
    )
    trial = current_flat + outer_lr * (merged - current_flat)
    rpp.apply_fragment(frag, trial.to(device), params)
    trial_loss, trial_by_batch = syncer_eval._losses(model, batches, compute_loss)
    rpp.apply_fragment(frag, current_flat.to(device), params)
    utility = base_loss - trial_loss
    utility_se = rpp._utility_se(base_by_batch, trial_by_batch)
    result = {
        f"{action}_utility": utility,
        f"{action}_utility_se": utility_se,
        f"{action}_negative": utility < 0.0,
        f"{action}_strict_negative": None if utility_se is None else utility + utility_se < 0.0,
        f"{action}_selected_count": len(selected),
        f"{action}_selected_mass": rpp._selected_mass(selected, all_candidates),
        f"{action}_outer_lr_multiplier": outer_lr,
    }
    out = {}
    for key, value in result.items():
        prefix = f"{action}_"
        assert key.startswith(prefix)
        out[f"{action}_{scope}_{key[len(prefix):]}"] = value
    if scope == "anchor" and args.include_anchor_batch_utilities:
        out[f"{action}_{scope}_batch_utilities"] = [
            float(base - trial) for base, trial in zip(base_by_batch, trial_by_batch)
        ]
    return out


def replay(args) -> tuple[list[dict], dict]:
    candidate_features = resolve_candidate_features(args)
    candidate_rows = read_jsonl(candidate_features)
    groups = rpp._group_rows(candidate_rows, min_candidates=args.min_candidates)
    if args.max_groups is not None and len(groups) > args.max_groups:
        rng = random.Random(args.sample_seed)
        groups = rng.sample(groups, args.max_groups)
        groups.sort(key=lambda g: (int(g[0]["pull_step"]), int(g[0]["fragment"])))

    deployable = tuple(args.deployable_actions)
    references = tuple(action for action in REFERENCE_ACTIONS if action in rpp.POLICIES)
    evaluated_actions = tuple(dict.fromkeys((*deployable, *references)))
    anchor_actions = deployable if args.anchor_deployable_only else evaluated_actions
    oracle_replay = load_oracle_replay(args.policy_replay)
    oracle_source = args.oracle_source
    if oracle_source == "auto":
        oracle_source = "precomputed" if oracle_replay else "compute"
    if oracle_source == "precomputed" and not oracle_replay:
        raise SystemExit("--oracle-source precomputed requires --policy-replay")

    anchor_rows, oracle_rows, manifest = data_splits(args)
    validate_manifest(manifest, require_disjoint=args.disjoint_anchor_oracle)

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
    anchor_batches = anchor_eval._batches_from_rows(
        anchor_rows,
        tokenizer=tokenizer,
        device=device,
        seq_len=args.seq_len,
        batch_size=args.anchor_batch_size,
        batches=args.anchor_batches,
        train_on=args.train_on,
    )
    oracle_batches = []
    if oracle_source == "compute":
        oracle_batches = anchor_eval._batches_from_rows(
            oracle_rows,
            tokenizer=tokenizer,
            device=device,
            seq_len=args.seq_len,
            batch_size=args.oracle_batch_size,
            batches=args.oracle_batches,
            train_on=args.train_on,
        )
    manifest["anchor_batches"] = len(anchor_batches)
    manifest["oracle_batches"] = len(oracle_batches) if oracle_batches else args.oracle_batches
    manifest["oracle_source"] = oracle_source
    manifest["candidate_features"] = str(candidate_features)
    manifest["deployable_actions"] = list(deployable)

    compute_loss = lambda logits, ids, w: sft_loss(logits, ids, args.loss_function, w)  # noqa: E731
    rng = random.Random(args.sample_seed)
    current_state_path: Path | None = None
    current_ckpt = None
    anchor_base_loss = 0.0
    anchor_base_by_batch: list[float] = []
    oracle_base_loss = 0.0
    oracle_base_by_batch: list[float] = []
    records = []
    was_training = model.training
    try:
        for idx, group in enumerate(groups, start=1):
            first = group[0]
            step = int(first["pull_step"])
            fid = int(first["fragment"])
            state_rel = str(first["capture_state_checkpoint"])
            state_path = rpp._resolve(args.capture_dir, state_rel)
            if current_state_path != state_path:
                current_ckpt = parse_checkpoint(state_path)
                syncer_eval._apply_checkpoint(current_ckpt, layout, params, device)
                anchor_base_loss, anchor_base_by_batch = syncer_eval._losses(
                    model, anchor_batches, compute_loss
                )
                if oracle_source == "compute":
                    oracle_base_loss, oracle_base_by_batch = syncer_eval._losses(
                        model, oracle_batches, compute_loss
                    )
                current_state_path = state_path
            assert current_ckpt is not None

            frag = layout.fragments[fid]
            current_flat = current_ckpt.fragments[fid][1]
            candidate_tensors = {
                (int(row["learner_id"]), fid): rpp._read_f32(
                    rpp._resolve(args.capture_dir, row["capture_candidate_f32"]), frag.numel
                )
                for row in group
            }
            replay_row = oracle_replay.get((step, fid))
            if oracle_source == "precomputed" and replay_row is None:
                raise SystemExit(f"missing oracle replay for step={step} fragment={fid}")

            record = {
                "schema": "action_probe_replay_v1",
                "seed": args.seed,
                "step": step,
                "fragment": fid,
                "state_checkpoint": state_rel,
                "candidate_count": len(group),
                "anchor_base_loss": anchor_base_loss,
                "score_field": args.score_field,
            }
            if oracle_source == "compute":
                record["oracle_base_loss"] = oracle_base_loss

            selections: dict[str, tuple[list[dict], float]] = {}
            for action in anchor_actions:
                selected, outer_lr = rpp._policy_candidates(
                    policy=action,
                    candidates=group,
                    score_field=args.score_field,
                    percentile=args.drop_percentile,
                    threshold=args.threshold,
                    min_selected_mass=args.min_selected_mass,
                    rng=rng,
                )
                selections[action] = (selected, outer_lr)
                record.update(
                    _eval_action_scope(
                        action=action,
                        scope="anchor",
                        selected=selected,
                        outer_lr=outer_lr,
                        current_flat=current_flat,
                        candidate_tensors=candidate_tensors,
                        model=model,
                        batches=anchor_batches,
                        compute_loss=compute_loss,
                        frag=frag,
                        params=params,
                        base_loss=anchor_base_loss,
                        base_by_batch=anchor_base_by_batch,
                        device=device,
                        all_candidates=group,
                        args=args,
                    )
                )
                record[f"{action}_selected_mass"] = record[f"{action}_anchor_selected_mass"]
                record[f"{action}_selected_count"] = record[f"{action}_anchor_selected_count"]

            if oracle_source == "precomputed":
                assert replay_row is not None
                _copy_oracle_metrics(record, replay_row, evaluated_actions)
            else:
                for action in evaluated_actions:
                    if action not in selections:
                        selected, outer_lr = rpp._policy_candidates(
                            policy=action,
                            candidates=group,
                            score_field=args.score_field,
                            percentile=args.drop_percentile,
                            threshold=args.threshold,
                            min_selected_mass=args.min_selected_mass,
                            rng=rng,
                        )
                        selections[action] = (selected, outer_lr)
                    selected, outer_lr = selections[action]
                    record.update(
                        _eval_action_scope(
                            action=action,
                            scope="oracle",
                            selected=selected,
                            outer_lr=outer_lr,
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
                            all_candidates=group,
                            args=args,
                        )
                    )

            token_oracle = float(record["token_weighted_oracle_utility"])
            oracle_positive = float(record["oracle_positive_oracle_utility"])
            oracle_topk = float(record["oracle_topk_oracle_utility"])
            record["oracle_positive_headroom"] = oracle_positive - token_oracle
            record["oracle_topk_headroom"] = oracle_topk - token_oracle

            best, best_anchor, best_margin = selected_action_by_anchor(record, deployable)
            record["chosen_policy_top1"] = best
            record["chosen_policy_top1_anchor_utility"] = best_anchor
            record["chosen_policy_top1_anchor_margin"] = best_margin
            record["chosen_policy_margin_gated"] = margin_gated_choice(
                record, deployable, args.margin
            )
            record["chosen_policy_risk_aware"] = risk_aware_choice(
                record, deployable, args.risk_threshold, args.risk_fallback
            )
            best_oracle = max(
                deployable, key=lambda action: float(record[f"{action}_oracle_utility"])
            )
            record["best_deployable_oracle_policy"] = best_oracle
            record["best_deployable_oracle_utility"] = float(
                record[f"{best_oracle}_oracle_utility"]
            )
            record["best_deployable_oracle_selected_mass"] = float(
                record[f"{best_oracle}_selected_mass"]
            )

            for policy_name, field in (
                ("action_probe_top1", "chosen_policy_top1"),
                ("action_probe_margin_gated", "chosen_policy_margin_gated"),
                ("action_probe_risk_aware", "chosen_policy_risk_aware"),
            ):
                chosen = str(record[field])
                metric = action_metric(record, chosen, "oracle")
                record[f"{policy_name}_chosen_action"] = chosen
                record[f"{policy_name}_oracle_utility"] = metric["utility"]
                record[f"{policy_name}_oracle_negative"] = metric["negative"]
                record[f"{policy_name}_oracle_strict_negative"] = metric["strict_negative"]
                record[f"{policy_name}_selected_mass"] = metric["selected_mass"]
                record[f"{policy_name}_selected_count"] = metric["selected_count"]
                record[f"{policy_name}_gain_vs_token"] = metric["utility"] - token_oracle

            records.append(record)
            sink = getattr(args, "_record_sink", None)
            if sink is not None:
                sink.write(json.dumps(jsonable(record), sort_keys=True, allow_nan=False) + "\n")
                sink.flush()
            if args.progress_every and (
                idx == 1 or idx % args.progress_every == 0 or idx == len(groups)
            ):
                print(
                    f"[action-probe] {idx}/{len(groups)} groups step={step} fragment={fid}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        model.train(was_training)
    return records, manifest


def summarize_choice(records: list[dict], choices: list[str]) -> dict:
    if len(records) != len(choices):
        raise ValueError("records and choices length mismatch")
    token_utils = [float(r["token_weighted_oracle_utility"]) for r in records]
    token_neg = mean([1.0 if r["token_weighted_oracle_negative"] else 0.0 for r in records])
    token_strict = mean(
        [
            1.0 if r["token_weighted_oracle_strict_negative"] else 0.0
            for r in records
            if r["token_weighted_oracle_strict_negative"] is not None
        ]
    )
    utilities = []
    gains = []
    negatives = []
    strict = []
    masses = []
    counts = []
    captured = []
    excluded = 0
    for record, action in zip(records, choices):
        if action == "best_deployable_oracle":
            utility = float(record["best_deployable_oracle_utility"])
            negative = utility < 0.0
            strict_negative = None
            mass = float(record["best_deployable_oracle_selected_mass"])
            count = float("nan")
        else:
            metric = action_metric(record, action, "oracle")
            utility = metric["utility"]
            negative = metric["negative"]
            strict_negative = metric["strict_negative"]
            mass = metric["selected_mass"]
            count = metric["selected_count"]
        token = float(record["token_weighted_oracle_utility"])
        oracle_positive = float(record["oracle_positive_oracle_utility"])
        utilities.append(utility)
        gains.append(utility - token)
        negatives.append(1.0 if negative else 0.0)
        if strict_negative is not None:
            strict.append(1.0 if strict_negative else 0.0)
        masses.append(mass)
        counts.append(count)
        denom = oracle_positive - token
        if denom > 0.0:
            captured.append((utility - token) / denom)
        else:
            excluded += 1
    neg_rate = mean(negatives)
    strict_rate = mean(strict)
    return {
        "groups": len(records),
        "mean_utility": mean(utilities),
        "utility_std": std(utilities),
        "mean_token_utility": mean(token_utils),
        "mean_gain_vs_token": mean(gains),
        "median_gain_vs_token": sorted(gains)[len(gains) // 2] if gains else float("nan"),
        "gain_positive_rate": mean([1.0 if gain > 0.0 else 0.0 for gain in gains]),
        "negative_rate": neg_rate,
        "negative_rate_relative_drop": (
            None if token_neg <= 0.0 else (token_neg - neg_rate) / token_neg
        ),
        "strict_negative_rate": strict_rate,
        "strict_negative_rate_relative_drop": (
            None if token_strict <= 0.0 else (token_strict - strict_rate) / token_strict
        ),
        "oracle_positive_headroom_captured": mean(captured),
        "headroom_excluded_fraction": excluded / len(records) if records else float("nan"),
        "selected_mass_mean": mean(masses),
        "selected_count_mean": mean(counts),
        "chosen_action_distribution": dict(sorted(Counter(choices).items())),
    }


def shuffled_action_control(records: list[dict], choices: list[str], *, trials: int, seed: int) -> dict:
    if not records or trials <= 0:
        return {}
    rng = random.Random(seed)
    metrics = []
    for _ in range(trials):
        shuffled = list(choices)
        rng.shuffle(shuffled)
        metrics.append(summarize_choice(records, shuffled))
    keys = (
        "mean_gain_vs_token",
        "negative_rate_relative_drop",
        "strict_negative_rate_relative_drop",
        "oracle_positive_headroom_captured",
        "selected_mass_mean",
    )
    return {key: mean([m[key] for m in metrics if m.get(key) is not None]) for key in keys}


def summarize(records: list[dict], deployable: tuple[str, ...], *, random_trials: int, seed: int) -> dict:
    if not records:
        raise SystemExit("no action-probe records")
    summaries = {}
    for action in deployable + REFERENCE_ACTIONS:
        if f"{action}_oracle_utility" in records[0]:
            summaries[action] = summarize_choice(records, [action] * len(records))

    best_fixed = max(deployable, key=lambda action: summaries[action]["mean_utility"])
    summaries["best_fixed_deployable"] = dict(summaries[best_fixed])
    summaries["best_fixed_deployable"]["fixed_action"] = best_fixed
    summaries["best_deployable_oracle"] = summarize_choice(
        records, ["best_deployable_oracle"] * len(records)
    )
    top1_choices = [str(r["chosen_policy_top1"]) for r in records]
    margin_choices = [str(r["chosen_policy_margin_gated"]) for r in records]
    risk_choices = [str(r["chosen_policy_risk_aware"]) for r in records]
    summaries["action_probe_top1"] = summarize_choice(records, top1_choices)
    summaries["action_probe_margin_gated"] = summarize_choice(records, margin_choices)
    summaries["action_probe_risk_aware"] = summarize_choice(records, risk_choices)
    summaries["random_top1_action_count"] = shuffled_action_control(
        records, top1_choices, trials=random_trials, seed=seed
    )
    main = max(
        ACTION_PROBE_POLICIES,
        key=lambda policy: summaries[policy]["mean_gain_vs_token"],
    )
    gates = {
        "main_policy": main,
        "mean_gain_ge_0.0005": summaries[main]["mean_gain_vs_token"] >= 0.0005,
        "negative_drop_ge_0.20": (
            summaries[main]["negative_rate_relative_drop"] is not None
            and summaries[main]["negative_rate_relative_drop"] >= 0.20
        ),
        "strict_drop_ge_0.20": (
            summaries[main]["strict_negative_rate_relative_drop"] is not None
            and summaries[main]["strict_negative_rate_relative_drop"] >= 0.20
        ),
        "headroom_captured_ge_0.40": (
            summaries[main]["oracle_positive_headroom_captured"] is not None
            and summaries[main]["oracle_positive_headroom_captured"] >= 0.40
        ),
        "selected_mass_ge_0.40": summaries[main]["selected_mass_mean"] >= 0.40,
        "beats_random_action_count": (
            summaries["random_top1_action_count"]
            and summaries[main]["mean_gain_vs_token"]
            > summaries["random_top1_action_count"]["mean_gain_vs_token"]
        ),
    }
    gates["single_seed_gate_pass"] = all(
        bool(v) for k, v in gates.items() if k != "main_policy"
    )
    return {
        "schema": "action_probe_summary_v1",
        "records": len(records),
        "seeds": sorted({int(r["seed"]) for r in records if r.get("seed") is not None}),
        "candidate_count_mean": mean([float(r["candidate_count"]) for r in records]),
        "oracle_positive_headroom_mean": mean(
            [float(r["oracle_positive_headroom"]) for r in records]
        ),
        "deployable_actions": list(deployable),
        "policies": summaries,
        "gates": gates,
    }


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    if abs(value) < 0.001 and value != 0.0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def to_markdown(summary: dict) -> str:
    lines = ["# Action-Probe Replay Summary", ""]
    lines.append(f"- Records: `{summary['records']}`")
    lines.append(f"- Seeds: `{summary['seeds']}`")
    lines.append(f"- Main policy: `{summary['gates']['main_policy']}`")
    lines.append(f"- Gate pass: `{summary['gates']['single_seed_gate_pass']}`")
    lines.append("")
    lines.append("| Policy | Gain vs token | Negative drop | Strict drop | Headroom captured | Selected mass |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    order = [
        "token_weighted",
        "best_fixed_deployable",
        "action_probe_top1",
        "action_probe_margin_gated",
        "action_probe_risk_aware",
        "best_deployable_oracle",
        "oracle_positive",
        "oracle_topk",
        "random_top1_action_count",
    ]
    for policy in order:
        row = summary["policies"].get(policy)
        if not row:
            continue
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                policy,
                fmt(row.get("mean_gain_vs_token"), 6),
                fmt(row.get("negative_rate_relative_drop")),
                fmt(row.get("strict_negative_rate_relative_drop")),
                fmt(row.get("oracle_positive_headroom_captured")),
                fmt(row.get("selected_mass_mean")),
            )
        )
    lines.append("")
    for policy in ACTION_PROBE_POLICIES:
        row = summary["policies"].get(policy)
        if row:
            lines.append(f"- `{policy}` choices: `{row['chosen_action_distribution']}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--candidate-features", type=Path)
    p.add_argument("--policy-replay", type=Path)
    p.add_argument("--split-manifest", type=Path)
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
    p.add_argument("--anchor-batches", type=int, default=2)
    p.add_argument("--anchor-batch-size", type=int, default=1)
    p.add_argument("--anchor-max-rows", type=int, default=64)
    p.add_argument("--oracle-batches", type=int, default=8)
    p.add_argument("--oracle-batch-size", type=int, default=1)
    p.add_argument("--oracle-max-rows", type=int, default=192)
    p.add_argument("--disjoint-anchor-oracle", action="store_true")
    p.add_argument("--oracle-source", choices=["auto", "precomputed", "compute"], default="auto")
    p.add_argument("--score-field", default="probe_grad_dot")
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--drop-percentile", type=float, default=0.30)
    p.add_argument("--min-selected-mass", type=float, default=0.35)
    p.add_argument("--margin", type=float, default=0.0005)
    p.add_argument("--risk-threshold", type=float, default=0.0)
    p.add_argument("--risk-fallback", choices=DEFAULT_DEPLOYABLE_ACTIONS, default="token_weighted")
    p.add_argument("--min-candidates", type=int, default=2)
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--include-extra-actions", action="store_true")
    p.add_argument("--deployable-actions", nargs="+", default=None)
    p.add_argument("--anchor-deployable-only", action="store_true")
    p.add_argument("--include-anchor-batch-utilities", action="store_true")
    p.add_argument("--random-trials", type=int, default=200)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--out-jsonl", required=True, type=Path)
    p.add_argument("--out-summary", required=True, type=Path)
    p.add_argument("--out-md", type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.deployable_actions is None:
        actions = list(DEFAULT_DEPLOYABLE_ACTIONS)
        if args.include_extra_actions:
            actions.extend(EXTRA_DEPLOYABLE_ACTIONS)
        args.deployable_actions = actions
    invalid = sorted(set(args.deployable_actions) - set(rpp.POLICIES))
    if invalid:
        raise SystemExit(f"unknown deployable actions: {invalid}")
    if any(action.startswith("oracle") or action.startswith("random") for action in args.deployable_actions):
        raise SystemExit("oracle/random actions cannot be deployable actions")
    args.seed = args.seed if args.seed is not None else infer_seed(args.capture_dir, args.features, row=None)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as sink:
        args._record_sink = sink
        records, manifest = replay(args)
    summary = summarize(
        records,
        tuple(args.deployable_actions),
        random_trials=args.random_trials,
        seed=args.sample_seed,
    )
    summary["split_manifest"] = manifest
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(to_markdown(summary))
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
