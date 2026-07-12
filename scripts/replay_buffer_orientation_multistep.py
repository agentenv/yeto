#!/usr/bin/env python3
"""Multi-step causal buffer-orientation intervention on captured outer commits.

The one-step intervention (replay_buffer_orientation.py) validated the exact
decomposition d_t = A_t g_t + d_t_perp but showed that ONE-STEP oracle loss
reverses the closed-loop harm ordering: with the real closed-loop cosine
c_t ~= -0.08, the aligned variant is locally BEST while mu=0.9 is harmful in
the real run, so the harm must be multi-step compounding. This script branches
at a captured commit t and rolls each buffer variant forward through N
consecutive same-fragment commits in OPEN LOOP: the merged deltas are the ones
recorded in the capture (the variant never regenerates learner work), while
the variant's own momentum buffer and fragment parameters evolve through the
exact production recursion

    b_k     = mu * b_{k-1} + delta_k
    theta_k = theta_{k-1} - lr * (delta_k + mu * b_k)

The held-out oracle loss is measured after the 1st and the Nth commit for
buffer variants of identical norm but different orientation at the branch
point (real, aligned, orthogonal, anti-aligned, random-rotated).

Prediction: over N commits the aligned amplification compounds (each commit
multiplies the shared g direction by A_k > 1 + mu while the orthogonal /
anti-aligned buffers inject less accumulated displacement), so the
MULTI-STEP loss should order with the accumulated displacement:
aligned > real > random_rotated ~ orthogonal > anti_aligned
(worst to best), restoring the closed-loop harm ordering that the one-step
loss reverses.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_script(name: str):
    import importlib.util

    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


orient = _load_script("replay_buffer_orientation")
bn = orient.bn
exact = orient.exact
buffered = orient.buffered
syncer_eval = orient.syncer_eval

from yeto.export import parse_checkpoint  # noqa: E402
from yeto.fragments import MERGE_AVG, build_layout  # noqa: E402
from yeto.learner import load_model_and_tokenizer, trainable_params  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402


REPLAY_SCHEMA = "buffer_orientation_multistep_replay_v1"
SUMMARY_SCHEMA = "buffer_orientation_multistep_summary_v1"
VARIANTS = orient.VARIANTS
BASELINE_VARIANT = orient.BASELINE_VARIANT
NORM_EPS = orient.NORM_EPS


def rollout(
    current: torch.Tensor,
    buffer: torch.Tensor,
    deltas: Sequence[torch.Tensor],
    outer_lr: float,
    mu: float,
    record_at: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor, list[dict], dict[int, torch.Tensor]]:
    """Chain the exact production Nesterov commit over captured deltas.

    Starting from parameters ``current`` and momentum buffer ``buffer``,
    applies for each delta (Rust convention g_k = delta_k = -merged_update)

        b_k     = mu * b_{k-1} + delta_k
        theta_k = theta_{k-1} - lr * (delta_k + mu * b_k)

    bit-matching bn._nesterov_trial at every commit. Inputs are never
    mutated. Returns (theta_N, b_N, per-commit stats, snapshots) where
    snapshots maps each 1-based commit index in ``record_at`` to a clone of
    theta_k after that commit.
    """

    if len(deltas) == 0:
        raise ValueError("rollout requires at least one delta")
    if buffer.shape != current.shape or buffer.ndim != 1:
        raise ValueError("current and buffer must be rank-1 tensors of equal shape")
    wanted = set(int(index) for index in record_at)
    unknown = wanted - set(range(1, len(deltas) + 1))
    if unknown:
        raise ValueError(f"record_at indices out of range: {sorted(unknown)}")

    theta = current
    b = buffer
    stats: list[dict] = []
    snapshots: dict[int, torch.Tensor] = {}
    cumulative_step_sq = 0.0
    for k, delta in enumerate(deltas, start=1):
        if delta.shape != current.shape:
            raise ValueError(f"delta {k} shape mismatch")
        delta_norm = float(delta.norm().item())
        if delta_norm < NORM_EPS:
            raise ValueError(f"delta {k} norm is numerically zero")
        c_k = float(torch.dot(b, delta).item()) / (delta_norm * delta_norm)
        buffer_norm = float(b.norm().item())
        next_b = b.mul(mu).add(delta)
        next_theta = theta - outer_lr * (delta + mu * next_b)
        step_norm = float((next_theta - theta).norm().item())
        cumulative_step_sq += step_norm * step_norm
        theta, b = next_theta, next_b
        stats.append(
            {
                "commit": k,
                "delta_norm": delta_norm,
                "buffer_norm": buffer_norm,
                "c": c_k,
                "aligned_gain": 1.0 + mu + mu * mu * c_k,
                "step_norm": step_norm,
                "cumulative_step_sq": cumulative_step_sq,
                "displacement_norm": float((theta - current).norm().item()),
            }
        )
        if k in wanted:
            snapshots[k] = theta.clone()
    return theta, b, stats, snapshots


def select_branch_groups(
    groups: Sequence[Sequence[dict]],
    branch_points: int,
    min_prior_rounds: int,
    rollout_commits: int,
) -> list[dict]:
    """Pick ~branch_points evenly spaced branch groups with full rollouts.

    A group is an eligible branch when its fragment has at least
    ``min_prior_rounds`` prior commits (so the reconstructed buffer is
    non-trivial) and at least ``rollout_commits - 1`` subsequent commits in
    the capture (so the open-loop rollout is fully covered by captured
    deltas). Returns one descriptor per branch with the group indices of all
    rollout commits (the branch group first).
    """

    if branch_points < 1:
        raise ValueError("branch_points must be positive")
    if min_prior_rounds < 1:
        raise ValueError("min_prior_rounds must be at least 1")
    if rollout_commits < 1:
        raise ValueError("rollout_commits must be positive")
    indices_by_fragment: dict[int, list[int]] = {}
    for index, group in enumerate(groups):
        fragment = int(group[0]["fragment"])
        indices_by_fragment.setdefault(fragment, []).append(index)

    eligible: list[dict] = []
    for index, group in enumerate(groups):
        fragment = int(group[0]["fragment"])
        sequence = indices_by_fragment[fragment]
        position = sequence.index(index)
        if position < min_prior_rounds:
            continue
        window = sequence[position : position + rollout_commits]
        if len(window) < rollout_commits:
            continue
        eligible.append({"branch_index": index, "rollout_indices": window})
    if not eligible:
        raise ValueError(
            f"no candidate group has {min_prior_rounds} prior rounds and "
            f"{rollout_commits - 1} subsequent rounds for its fragment"
        )
    if len(eligible) <= branch_points:
        return eligible
    positions = [
        round(rank * (len(eligible) - 1) / (branch_points - 1))
        if branch_points > 1
        else 0
        for rank in range(branch_points)
    ]
    picked = sorted({int(position) for position in positions})
    return [eligible[position] for position in picked]


def rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks for ties)."""

    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("rank correlation needs two equal-length series")

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: float(values[index]))
        out = [0.0] * len(values)
        start = 0
        while start < len(order):
            stop = start
            while (
                stop + 1 < len(order)
                and float(values[order[stop + 1]]) == float(values[order[start]])
            ):
                stop += 1
            rank = (start + stop) / 2.0 + 1.0
            for position in range(start, stop + 1):
                out[order[position]] = rank
            start = stop + 1
        return out

    rx = ranks(xs)
    ry = ranks(ys)
    mx = exact.mean(rx)
    my = exact.mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def _group_sort_key(group: Sequence[dict]) -> tuple[int, int]:
    return (int(group[0]["step"]), int(group[0]["fragment"]))


def _ordering_rate(records: list[dict], key: str, worse: str, better: str) -> float:
    return exact.mean(
        [
            float(record["variants"][worse][key])
            > float(record["variants"][better][key])
            for record in records
        ]
    )


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"schema": SUMMARY_SCHEMA, "records": 0, "variants": {}}
    per_variant = {}
    final_by_variant = {
        name: [float(record["variants"][name]["oracle_loss"]) for record in records]
        for name in VARIANTS
    }
    first_by_variant = {
        name: [
            float(record["variants"][name]["first_commit_oracle_loss"])
            for record in records
        ]
        for name in VARIANTS
    }
    real_final = final_by_variant[BASELINE_VARIANT]
    real_first = first_by_variant[BASELINE_VARIANT]
    branch_losses = [float(record["oracle_branch_loss"]) for record in records]
    for name in VARIANTS:
        final = final_by_variant[name]
        first = first_by_variant[name]
        vs_real = [real - loss for real, loss in zip(real_final, final)]
        vs_real_first = [real - loss for real, loss in zip(real_first, first)]
        utilities = [base - loss for base, loss in zip(branch_losses, final)]
        per_variant[name] = {
            "records": len(records),
            "mean_final_oracle_loss": exact.mean(final),
            "mean_final_utility_vs_branch": exact.mean(utilities),
            "mean_final_gain_vs_real": exact.mean(vs_real),
            "final_gain_vs_real_se": (
                0.0
                if len(vs_real) < 2
                else exact.std(vs_real) / math.sqrt(len(vs_real))
            ),
            "final_gain_vs_real_positive_rate": exact.mean(
                [value > 0.0 for value in vs_real]
            ),
            "mean_first_commit_oracle_loss": exact.mean(first),
            "mean_first_commit_gain_vs_real": exact.mean(vs_real_first),
            "first_commit_gain_vs_real_positive_rate": exact.mean(
                [value > 0.0 for value in vs_real_first]
            ),
            "mean_displacement_norm": exact.mean(
                [
                    float(record["variants"][name]["displacement_norm"])
                    for record in records
                ]
            ),
            "mean_cumulative_step_sq": exact.mean(
                [
                    float(record["variants"][name]["cumulative_step_sq"])
                    for record in records
                ]
            ),
            "mean_branch_aligned_gain": exact.mean(
                [
                    float(record["variants"][name]["branch_aligned_gain"])
                    for record in records
                ]
            ),
            "mean_rollout_aligned_gain": exact.mean(
                [
                    float(record["variants"][name]["mean_rollout_aligned_gain"])
                    for record in records
                ]
            ),
        }
    displacement_loss_rank_correlations = [
        rank_correlation(
            [float(record["variants"][name]["displacement_norm"]) for name in VARIANTS],
            [float(record["variants"][name]["oracle_loss"]) for name in VARIANTS],
        )
        for record in records
    ]
    orderings = {
        "final_aligned_worse_than_real_rate": _ordering_rate(
            records, "oracle_loss", "aligned", "real"
        ),
        "final_aligned_worse_than_orthogonal_rate": _ordering_rate(
            records, "oracle_loss", "aligned", "orthogonal"
        ),
        "final_aligned_worse_than_anti_aligned_rate": _ordering_rate(
            records, "oracle_loss", "aligned", "anti_aligned"
        ),
        "final_orthogonal_worse_than_anti_aligned_rate": _ordering_rate(
            records, "oracle_loss", "orthogonal", "anti_aligned"
        ),
        "final_random_orthogonal_absolute_gap": exact.mean(
            [
                abs(
                    final_by_variant["random_rotated"][index]
                    - final_by_variant["orthogonal"][index]
                )
                for index in range(len(records))
            ]
        ),
        "first_commit_aligned_worse_than_real_rate": _ordering_rate(
            records, "first_commit_oracle_loss", "aligned", "real"
        ),
        "first_commit_aligned_worse_than_anti_aligned_rate": _ordering_rate(
            records, "first_commit_oracle_loss", "aligned", "anti_aligned"
        ),
        "mean_displacement_loss_rank_correlation": exact.mean(
            displacement_loss_rank_correlations
        ),
        "displacement_loss_rank_correlation_positive_rate": exact.mean(
            [value > 0.0 for value in displacement_loss_rank_correlations]
        ),
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "records": len(records),
        "seeds": sorted({int(record["seed"]) for record in records}),
        "branch_steps": [int(record["step"]) for record in records],
        "rollout_commits": int(records[0]["rollout_commits"]),
        "mean_branch_oracle_loss": exact.mean(branch_losses),
        "mean_branch_buffer_norm": exact.mean(
            [float(record["buffer_norm"]) for record in records]
        ),
        "mean_branch_delta_norm": exact.mean(
            [float(record["delta_norm"]) for record in records]
        ),
        "mean_branch_real_c_t": exact.mean(
            [float(record["c_t"]) for record in records]
        ),
        "max_buffer_checkpoint_relative_error": max(
            float(record["buffer_checkpoint_relative_error"]) for record in records
        ),
        "variants": per_variant,
        "orderings": orderings,
        "prediction": {
            "theory": (
                "one-step loss reverses the closed-loop harm ordering (aligned "
                "locally best at c_t ~= -0.08); over N open-loop commits the "
                "aligned buffer compounds the shared-direction amplification "
                "A_k = 1 + mu + mu^2 c_k while orthogonal/anti-aligned buffers "
                "accumulate less displacement, so multi-step loss should order "
                "with accumulated displacement: aligned > real > "
                "random_rotated ~ orthogonal > anti_aligned (worst to best)"
            ),
            "expected_final_loss_order_worst_to_best": [
                "aligned",
                "real",
                "random_rotated",
                "orthogonal",
                "anti_aligned",
            ],
        },
    }


def replay(args) -> list[dict]:
    root = args.capture_dir
    if args.expected_candidates:
        groups = exact.validate_candidate_groups(
            buffered._read_jsonl(root / "index.jsonl"), args.expected_candidates
        )
    else:
        groups = buffered._group_rows(buffered._read_jsonl(root / "index.jsonl"), 1)
    groups = sorted(groups, key=_group_sort_key)
    if not groups:
        raise SystemExit("capture contains no candidate groups")
    try:
        descriptors = select_branch_groups(
            groups, args.branch_points, args.min_prior_rounds, args.rollout_commits
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    needed_indices = sorted(
        {index for descriptor in descriptors for index in descriptor["rollout_indices"]}
    )
    needed = set(needed_indices)
    branch_by_index = {
        descriptor["branch_index"]: descriptor for descriptor in descriptors
    }
    last_needed = needed_indices[-1]

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
    try:
        oracle = exact.build_row_panels(
            args.oracle_data,
            tokenizer,
            seq_len=args.seq_len,
            panel_count=args.oracle_panels,
            blocks_per_panel=args.oracle_blocks_per_panel,
            max_rows=args.oracle_max_rows,
            train_on=args.train_on,
            device=device,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    replay_config = {
        "schema": REPLAY_SCHEMA,
        "seed": args.seed,
        "capture_dir": str(args.capture_dir.expanduser().resolve()),
        "model": args.model,
        "outer_optimizer": "nesterov",
        "outer_lr": args.outer_lr,
        "outer_momentum": args.outer_momentum,
        "delta_correction": args.delta_correction,
        "buffer_source": "recursion_replay",
        "rollout_mode": "open_loop_captured_deltas",
        "variants": list(VARIANTS),
        "branch_points": args.branch_points,
        "rollout_commits": args.rollout_commits,
        "min_prior_rounds": args.min_prior_rounds,
        "expected_candidates": args.expected_candidates,
        "layout": {
            "tuning": args.tuning,
            "fragments": args.fragments,
            "fragment_pattern": args.fragment_pattern,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_targets": args.lora_targets,
        },
        "oracle_metadata": oracle["metadata"],
    }
    replay_config_sha256 = exact._config_sha256(replay_config)
    args._replay_config = replay_config
    args._replay_config_sha256 = replay_config_sha256

    compute_loss = lambda logits, ids, weights: sft_loss(  # noqa: E731
        logits, ids, args.loss_function, weights
    )

    # --- phase 1: replay the real buffer recursion, collect needed deltas ---
    buffers: dict[int, torch.Tensor] = {}
    checkpoint_cache: tuple[str, object] | None = None
    deltas_by_index: dict[int, torch.Tensor] = {}
    delta_meta_by_index: dict[int, dict] = {}
    branch_states: dict[int, dict] = {}
    for group_index, group in enumerate(groups):
        if group_index > last_needed:
            break
        first = group[0]
        state_checkpoint = str(first["state_checkpoint"])
        if checkpoint_cache is None or checkpoint_cache[0] != state_checkpoint:
            checkpoint_cache = (
                state_checkpoint,
                parse_checkpoint(buffered._resolve(root, state_checkpoint)),
            )
        checkpoint = checkpoint_cache[1]
        fragment_id = int(first["fragment"])
        frag = layout.fragments[fragment_id]
        current = checkpoint.fragments[fragment_id][1]
        stored_momentum = checkpoint.fragments[fragment_id][2]
        current_version = int(checkpoint.fragments[fragment_id][0])
        if fragment_id not in buffers:
            buffers[fragment_id] = torch.zeros_like(current)
        buffer = buffers[fragment_id]

        buffer_error = float((buffer - stored_momentum).norm().item()) / max(
            float(stored_momentum.norm().item()), NORM_EPS
        )
        if float(stored_momentum.norm().item()) < NORM_EPS:
            buffer_error = float(buffer.norm().item())

        merge_momentum = (
            buffer if args.delta_correction == "heloco" else torch.zeros_like(buffer)
        )
        candidates = []
        for row in sorted(group, key=lambda item: int(item["learner_id"])):
            tensor = buffered._read_f32(
                buffered._resolve(root, row["candidate_f32"]), frag.numel
            )
            if not bool(torch.isfinite(tensor).all()):
                raise SystemExit(
                    f"step={first['step']} fragment={fragment_id}: "
                    "candidate contains NaN or Inf"
                )
            candidates.append(
                buffered._candidate(
                    row, tensor, current, merge_momentum, current_version
                )
            )
        merged_update = bn._production_merge_update(candidates, merge_momentum, frag)
        if not bool(torch.isfinite(merged_update).all()):
            raise SystemExit(
                f"step={first['step']} fragment={fragment_id}: "
                "merged group delta contains NaN or Inf"
            )
        delta = -merged_update

        if group_index in branch_by_index:
            if buffer_error > args.max_buffer_relative_error:
                raise SystemExit(
                    f"step={first['step']} fragment={fragment_id}: reconstructed "
                    f"buffer diverged from the stored checkpoint momentum "
                    f"(relative_error={buffer_error:.3e} > "
                    f"{args.max_buffer_relative_error:.3e}); the capture likely "
                    "misses sync rounds for this fragment"
                )
            branch_states[group_index] = {
                "group": group,
                "state_checkpoint": state_checkpoint,
                "buffer": buffer.clone(),
                "buffer_error": buffer_error,
                "candidate_weights": [
                    bn.soft._candidate_weight(row)
                    for row in sorted(group, key=lambda item: int(item["learner_id"]))
                ],
            }
        if group_index in needed:
            deltas_by_index[group_index] = delta.clone()
            delta_meta_by_index[group_index] = {
                "group_index": group_index,
                "step": int(first["step"]),
                "fragment": fragment_id,
                "delta_norm": float(delta.norm().item()),
                "buffer_checkpoint_relative_error": buffer_error,
            }

        buffers[fragment_id] = buffer.mul(args.outer_momentum).add(delta)

    # --- phase 2: branch interventions and open-loop rollouts ---
    records = []
    was_training = model.training
    model.eval()
    checkpoint_cache = None
    try:
        for descriptor in descriptors:
            branch_index = descriptor["branch_index"]
            state = branch_states[branch_index]
            if (
                checkpoint_cache is None
                or checkpoint_cache[0] != state["state_checkpoint"]
            ):
                checkpoint_cache = (
                    state["state_checkpoint"],
                    parse_checkpoint(
                        buffered._resolve(root, state["state_checkpoint"])
                    ),
                )
            checkpoint = checkpoint_cache[1]
            records.append(
                _intervene(
                    args,
                    model,
                    params,
                    layout,
                    oracle,
                    compute_loss,
                    checkpoint,
                    descriptor,
                    state,
                    deltas_by_index,
                    delta_meta_by_index,
                    device,
                )
            )
            args._sink.write(
                json.dumps(
                    exact.jsonable(records[-1]), sort_keys=True, allow_nan=False
                )
                + "\n"
            )
            args._sink.flush()
            if args.progress_every and (
                len(records) == 1 or len(records) % args.progress_every == 0
            ):
                print(
                    f"[buffer-orientation-multistep] branches={len(records)}/"
                    f"{len(descriptors)} step={records[-1]['step']} "
                    f"fragment={records[-1]['fragment']} "
                    f"buffer_rel_err={state['buffer_error']:.2e}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        model.train(was_training)
    return records


def _intervene(
    args,
    model,
    params,
    layout,
    oracle,
    compute_loss,
    checkpoint,
    descriptor,
    state,
    deltas_by_index,
    delta_meta_by_index,
    device,
) -> dict:
    group = state["group"]
    first = group[0]
    fragment_id = int(first["fragment"])
    frag = layout.fragments[fragment_id]
    current = checkpoint.fragments[fragment_id][1]
    mu = args.outer_momentum
    buffer = state["buffer"]
    rollout_indices = descriptor["rollout_indices"]
    deltas = [deltas_by_index[index] for index in rollout_indices]
    branch_delta = deltas[0]

    try:
        geometry = orient.buffer_geometry(buffer, branch_delta, mu)
        generator = torch.Generator(device="cpu")
        commit_seed = orient.variant_seed(args.seed, int(first["step"]), fragment_id)
        generator.manual_seed(commit_seed)
        variants = orient.build_buffer_variants(buffer, branch_delta, generator)
    except ValueError as exc:
        raise SystemExit(
            f"step={first['step']} fragment={fragment_id}: {exc}"
        ) from exc

    syncer_eval._apply_checkpoint(checkpoint, layout, params, device)
    oracle_branch_loss, oracle_branch_panel_losses = syncer_eval._losses(
        model, oracle["panels"], compute_loss
    )

    results = {}
    for name in VARIANTS:
        theta_final, buffer_final, stats, snapshots = rollout(
            current,
            variants[name],
            deltas,
            args.outer_lr,
            mu,
            record_at=(1,) if len(deltas) > 1 else (),
        )
        final_loss, final_panel_losses = exact.losses_for_trial(
            model,
            oracle["panels"],
            compute_loss,
            frag,
            params,
            current,
            theta_final,
            device,
        )
        if len(deltas) > 1:
            first_loss, first_panel_losses = exact.losses_for_trial(
                model,
                oracle["panels"],
                compute_loss,
                frag,
                params,
                current,
                snapshots[1],
                device,
            )
        else:
            first_loss, first_panel_losses = final_loss, final_panel_losses
        results[name] = {
            "stats": stats,
            "final_buffer_norm": float(buffer_final.norm().item()),
            "loss": final_loss,
            "panel_losses": final_panel_losses,
            "first_loss": first_loss,
            "first_panel_losses": first_panel_losses,
        }

    real_panel_losses = results[BASELINE_VARIANT]["panel_losses"]
    real_first_panel_losses = results[BASELINE_VARIANT]["first_panel_losses"]
    variant_records = {}
    for name in VARIANTS:
        variant_geometry = orient.buffer_geometry(variants[name], branch_delta, mu)
        stats = results[name]["stats"]
        variant_records[name] = {
            "name": name,
            "buffer_norm": variant_geometry["buffer_norm"],
            "branch_cosine_to_delta": variant_geometry["cosine_to_delta"],
            "branch_c_t": variant_geometry["c_t"],
            "branch_aligned_gain": variant_geometry["aligned_gain"],
            "rollout_stats": stats,
            "mean_rollout_aligned_gain": exact.mean(
                [float(item["aligned_gain"]) for item in stats]
            ),
            "displacement_norm": float(stats[-1]["displacement_norm"]),
            "cumulative_step_sq": float(stats[-1]["cumulative_step_sq"]),
            "final_buffer_norm": results[name]["final_buffer_norm"],
            "oracle_loss": results[name]["loss"],
            "oracle_panel_losses": results[name]["panel_losses"],
            "oracle_utility_vs_branch": oracle_branch_loss - results[name]["loss"],
            "first_commit_oracle_loss": results[name]["first_loss"],
            "paired_vs_branch": orient.paired_panel_stats(
                oracle_branch_panel_losses, results[name]["panel_losses"]
            ),
            "paired_vs_real": orient.paired_panel_stats(
                real_panel_losses, results[name]["panel_losses"]
            ),
            "first_commit_paired_vs_real": orient.paired_panel_stats(
                real_first_panel_losses, results[name]["first_panel_losses"]
            ),
        }

    return {
        "schema": REPLAY_SCHEMA,
        "replay_config_sha256": args._replay_config_sha256,
        "seed": args.seed,
        "group_index": descriptor["branch_index"],
        "state_checkpoint": state["state_checkpoint"],
        "step": int(first["step"]),
        "fragment": fragment_id,
        "merge_mode": "avg" if frag.merge_mode == MERGE_AVG else "rda",
        "outer_optimizer": "nesterov",
        "outer_lr": args.outer_lr,
        "outer_momentum": mu,
        "delta_correction": args.delta_correction,
        "rollout_commits": args.rollout_commits,
        "rollout_mode": "open_loop_captured_deltas",
        "rollout_groups": [
            delta_meta_by_index[index] for index in rollout_indices
        ],
        "candidate_count": len(group),
        "candidate_learner_ids": sorted(int(row["learner_id"]) for row in group),
        "candidate_weights": state["candidate_weights"],
        "buffer_source": "recursion_replay",
        "buffer_checkpoint_relative_error": state["buffer_error"],
        "buffer_norm": geometry["buffer_norm"],
        "delta_norm": geometry["delta_norm"],
        "buffer_cosine_to_delta": geometry["cosine_to_delta"],
        "c_t": geometry["c_t"],
        "r_t": geometry["r_t"],
        "aligned_gain": geometry["aligned_gain"],
        "transverse_ratio": geometry["transverse_ratio"],
        "variant_seed": commit_seed,
        "oracle_panels_sha256": oracle["metadata"]["panel_tensors_sha256"],
        "oracle_branch_loss": oracle_branch_loss,
        "oracle_branch_panel_losses": oracle_branch_panel_losses,
        "variants": variant_records,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--oracle-data", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--tuning", choices=["lora", "full"], default="lora")
    parser.add_argument("--lora-r", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument(
        "--lora-targets",
        choices=["auto", "attention", "all-linear"],
        default="auto",
    )
    parser.add_argument("--fragments", type=int, default=4)
    parser.add_argument(
        "--fragment-pattern", choices=["binpack", "strided"], default="binpack"
    )
    parser.add_argument("--loss-function", default="cross_entropy")
    parser.add_argument("--train-on", choices=["assistant", "all"], default="assistant")
    parser.add_argument("--oracle-panels", type=int, default=8)
    parser.add_argument("--oracle-blocks-per-panel", type=int, default=2)
    parser.add_argument("--oracle-max-rows", type=int, default=256)
    parser.add_argument("--outer-lr", type=float, default=0.175)
    parser.add_argument("--outer-momentum", type=float, default=0.9)
    parser.add_argument(
        "--delta-correction",
        choices=["heloco", "none"],
        default="heloco",
        help="Match the production merge used when the capture was recorded.",
    )
    parser.add_argument("--branch-points", type=int, default=6)
    parser.add_argument("--rollout-commits", type=int, default=8)
    parser.add_argument("--min-prior-rounds", type=int, default=1)
    parser.add_argument(
        "--expected-candidates",
        type=int,
        default=0,
        help="If positive, require every group to have exactly this many candidates.",
    )
    parser.add_argument("--max-buffer-relative-error", type=float, default=1e-3)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cuda is required for Qwen 9B captures; cpu works for SmolLM2.",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    args = parser.parse_args(argv)

    args.seed = (
        args.seed if args.seed is not None else exact.infer_seed(args.capture_dir)
    )
    if args.seed is None:
        parser.error("could not infer seed; pass --seed")
    if not math.isfinite(args.outer_lr) or args.outer_lr <= 0.0:
        parser.error("--outer-lr must be finite and positive")
    if not 0.0 < args.outer_momentum < 1.0:
        parser.error("--outer-momentum must be in (0, 1)")
    if args.seq_len <= 1:
        parser.error("--seq-len must be greater than 1")
    if args.branch_points < 1:
        parser.error("--branch-points must be positive")
    if args.rollout_commits < 1:
        parser.error("--rollout-commits must be positive")
    if args.min_prior_rounds < 1:
        parser.error("--min-prior-rounds must be at least 1")
    if args.expected_candidates < 0:
        parser.error("--expected-candidates must be >= 0")
    if args.oracle_panels < 2:
        parser.error("--oracle-panels must be at least 2")
    if args.oracle_blocks_per_panel < 1:
        parser.error("--oracle-blocks-per-panel must be positive")
    if args.oracle_max_rows < args.oracle_panels:
        parser.error("--oracle-max-rows must be at least --oracle-panels")
    if (
        not math.isfinite(args.max_buffer_relative_error)
        or args.max_buffer_relative_error <= 0.0
    ):
        parser.error("--max-buffer-relative-error must be finite and positive")
    if args.out_jsonl.resolve() == args.out_summary.resolve():
        parser.error("--out-jsonl and --out-summary must be different paths")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as sink:
        args._sink = sink
        records = replay(args)
    summary = summarize(records)
    summary["replay_config"] = args._replay_config
    summary["replay_config_sha256"] = args._replay_config_sha256
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(exact.jsonable(summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    args.out_summary.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
