#!/usr/bin/env python3
"""Causal buffer-orientation intervention on captured outer commits.

For ~N evenly spaced commits per capture, the momentum buffer b_{t-1} is
reconstructed by replaying the exact production buffer recursion
b_s = mu * b_{s-1} + delta_s over every prior candidate group, validated
against the momentum stored in each captured state checkpoint. At each
selected commit the same merged delta is applied through one outer Nesterov
step d_t = delta_t + mu * (mu * b + delta_t) with buffer variants of
identical norm but different orientation (real, fully aligned with the
pseudo-gradient, orthogonalized, anti-aligned, random-rotated), and the
one-step held-out oracle loss of every variant is measured against the
no-step baseline with a paired panel evaluation.

Per the exact decomposition d_t = A_t g_t + d_t_perp with
A_t = 1 + mu + mu^2 c_t and |d_t_perp| = mu^2 |b - c_t g_t|, the variants
isolate the aligned gain A_t (aligned > real > orthogonal ~ random >
anti-aligned) at fixed buffer norm.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Sequence

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
exact = _load_script("replay_exact_lr_probe")
buffered = bn.buffered
syncer_eval = bn.syncer_eval

from yeto.export import parse_checkpoint  # noqa: E402
from yeto.fragments import MERGE_AVG, build_layout  # noqa: E402
from yeto.learner import load_model_and_tokenizer, trainable_params  # noqa: E402
from yeto.losses import sft_loss  # noqa: E402


REPLAY_SCHEMA = "buffer_orientation_replay_v1"
SUMMARY_SCHEMA = "buffer_orientation_summary_v1"
VARIANTS = ("real", "aligned", "orthogonal", "anti_aligned", "random_rotated")
BASELINE_VARIANT = "real"
NORM_EPS = 1e-12


def variant_seed(seed: int, step: int, fragment: int) -> int:
    """Derive a deterministic per-commit seed independent of commit order."""

    payload = f"buffer_orientation:{int(seed)}:{int(step)}:{int(fragment)}"
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def buffer_geometry(buffer: torch.Tensor, delta: torch.Tensor, mu: float) -> dict:
    """Exact one-commit decomposition of the Nesterov direction.

    Uses the Rust gradient convention g_t = delta_t = -merged_update, so
    c_t = <b, g_t> / |g_t|^2, A_t = 1 + mu + mu^2 c_t and the transverse
    component of d_t has norm mu^2 |b - c_t g_t|.
    """

    delta_norm = float(delta.norm().item())
    if delta_norm < NORM_EPS:
        raise ValueError("merged delta norm is numerically zero")
    buffer_norm = float(buffer.norm().item())
    c_t = float(torch.dot(buffer, delta).item()) / (delta_norm * delta_norm)
    residual = buffer - c_t * delta
    r_t = float(residual.norm().item()) / delta_norm
    return {
        "buffer_norm": buffer_norm,
        "delta_norm": delta_norm,
        "cosine_to_delta": (
            0.0
            if buffer_norm < NORM_EPS
            else c_t * delta_norm / buffer_norm
        ),
        "c_t": c_t,
        "r_t": r_t,
        "aligned_gain": 1.0 + mu + mu * mu * c_t,
        "transverse_ratio": mu * mu * r_t,
    }


PARALLEL_RESIDUAL_RTOL = 1e-6


def _project_out(vector: torch.Tensor, unit_delta: torch.Tensor) -> torch.Tensor:
    """Remove the unit_delta component twice to defeat f32 cancellation."""

    out = vector - torch.dot(vector, unit_delta) * unit_delta
    return out - torch.dot(out, unit_delta) * unit_delta


def _orthogonal_from(
    buffer: torch.Tensor, unit_delta: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Return a unit vector orthogonal to unit_delta near the buffer plane."""

    buffer_norm = float(buffer.norm().item())
    residual = _project_out(buffer, unit_delta)
    residual_norm = float(residual.norm().item())
    if residual_norm >= PARALLEL_RESIDUAL_RTOL * buffer_norm:
        return residual / residual_norm
    # Degenerate case: the buffer is numerically parallel to the delta, so
    # the residual is pure rounding noise (still delta-dominated); draw the
    # orthogonal direction deterministically instead.
    fallback = _project_out(
        torch.randn(
            buffer.shape, generator=generator, dtype=buffer.dtype, device=buffer.device
        ),
        unit_delta,
    )
    fallback_norm = float(fallback.norm().item())
    if fallback_norm < NORM_EPS:
        raise ValueError("could not construct an orthogonal buffer direction")
    return fallback / fallback_norm


def build_buffer_variants(
    buffer: torch.Tensor, delta: torch.Tensor, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Build the five equal-norm buffer orientations for one commit.

    delta is the Rust-convention pseudo-gradient g_t = -merged_update. All
    variants share the exact norm |b|; only the orientation changes. The
    random draw order is fixed (rotation first, orthogonal fallback second)
    so results are reproducible from the generator seed alone.
    """

    if buffer.shape != delta.shape or buffer.ndim != 1:
        raise ValueError("buffer and delta must be rank-1 tensors of equal shape")
    delta_norm = float(delta.norm().item())
    if delta_norm < NORM_EPS:
        raise ValueError("merged delta norm is numerically zero")
    buffer_norm = float(buffer.norm().item())
    if buffer_norm < NORM_EPS:
        raise ValueError("buffer norm is numerically zero; nothing to reorient")
    unit_delta = delta / delta_norm

    rotated = torch.randn(
        buffer.shape, generator=generator, dtype=buffer.dtype, device=buffer.device
    )
    rotated_norm = float(rotated.norm().item())
    if rotated_norm < NORM_EPS:
        raise ValueError("random rotation draw collapsed to zero")
    orthogonal_unit = _orthogonal_from(buffer, unit_delta, generator)

    return {
        "real": buffer.clone(),
        "aligned": unit_delta * buffer_norm,
        "orthogonal": orthogonal_unit * buffer_norm,
        "anti_aligned": unit_delta * (-buffer_norm),
        "random_rotated": rotated * (buffer_norm / rotated_norm),
    }


def nesterov_direction(
    delta: torch.Tensor, buffer: torch.Tensor, mu: float
) -> torch.Tensor:
    """Exact two-term form d_t = (1 + mu) delta_t + mu^2 b_{t-1}."""

    return (1.0 + mu) * delta + mu * mu * buffer


def select_commit_groups(
    groups: Sequence[Sequence[dict]], commits: int, min_prior_rounds: int
) -> list[int]:
    """Pick ~commits evenly spaced group indices with enough fragment history."""

    if commits < 1:
        raise ValueError("commits must be positive")
    if min_prior_rounds < 1:
        raise ValueError("min_prior_rounds must be at least 1")
    prior_by_fragment: dict[int, int] = {}
    eligible = []
    for index, group in enumerate(groups):
        fragment = int(group[0]["fragment"])
        prior = prior_by_fragment.get(fragment, 0)
        if prior >= min_prior_rounds:
            eligible.append(index)
        prior_by_fragment[fragment] = prior + 1
    if not eligible:
        raise ValueError(
            "no candidate group has at least "
            f"{min_prior_rounds} prior rounds for its fragment"
        )
    if len(eligible) <= commits:
        return eligible
    positions = [
        round(rank * (len(eligible) - 1) / (commits - 1)) if commits > 1 else 0
        for rank in range(commits)
    ]
    return sorted({eligible[position] for position in positions})


def paired_panel_stats(
    reference_losses: Sequence[float], variant_losses: Sequence[float]
) -> dict:
    """Per-panel paired differences (reference - variant; positive = better)."""

    reference = [float(value) for value in reference_losses]
    variant = [float(value) for value in variant_losses]
    if not reference or len(reference) != len(variant):
        raise ValueError("paired loss series must be non-empty and equal length")
    if not all(math.isfinite(value) for value in reference + variant):
        raise ValueError("paired loss series contains a non-finite value")
    gains = [left - right for left, right in zip(reference, variant)]
    gain = exact.mean(gains)
    se = 0.0 if len(gains) < 2 else exact.std(gains) / math.sqrt(len(gains))
    return {
        "mean_gain": gain,
        "se": se,
        "win_rate": sum(value > 0.0 for value in gains) / len(gains),
        "panels": len(gains),
    }


def _group_sort_key(group: Sequence[dict]) -> tuple[int, int]:
    return (int(group[0]["step"]), int(group[0]["fragment"]))


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"schema": SUMMARY_SCHEMA, "records": 0, "variants": {}}
    per_variant = {}
    losses_by_variant = {
        name: [
            float(record["variants"][name]["oracle_loss"]) for record in records
        ]
        for name in VARIANTS
    }
    real_losses = losses_by_variant[BASELINE_VARIANT]
    no_step = [float(record["oracle_current_loss"]) for record in records]
    for name in VARIANTS:
        losses = losses_by_variant[name]
        utilities = [base - loss for base, loss in zip(no_step, losses)]
        vs_real = [real - loss for real, loss in zip(real_losses, losses)]
        per_variant[name] = {
            "records": len(records),
            "mean_oracle_loss": exact.mean(losses),
            "mean_oracle_utility": exact.mean(utilities),
            "utility_positive_rate": exact.mean(
                [value > 0.0 for value in utilities]
            ),
            "mean_gain_vs_real": exact.mean(vs_real),
            "gain_vs_real_se": (
                0.0
                if len(vs_real) < 2
                else exact.std(vs_real) / math.sqrt(len(vs_real))
            ),
            "gain_vs_real_positive_rate": exact.mean(
                [value > 0.0 for value in vs_real]
            ),
            "mean_aligned_gain": exact.mean(
                [
                    float(record["variants"][name]["aligned_gain"])
                    for record in records
                ]
            ),
            "mean_cosine_to_delta": exact.mean(
                [
                    float(record["variants"][name]["cosine_to_delta"])
                    for record in records
                ]
            ),
            "mean_transverse_ratio": exact.mean(
                [
                    float(record["variants"][name]["transverse_ratio"])
                    for record in records
                ]
            ),
        }
    orderings = {
        "aligned_worse_than_orthogonal_rate": exact.mean(
            [
                losses_by_variant["aligned"][index]
                > losses_by_variant["orthogonal"][index]
                for index in range(len(records))
            ]
        ),
        "orthogonal_worse_than_anti_aligned_rate": exact.mean(
            [
                losses_by_variant["orthogonal"][index]
                > losses_by_variant["anti_aligned"][index]
                for index in range(len(records))
            ]
        ),
        "aligned_worse_than_real_rate": exact.mean(
            [
                losses_by_variant["aligned"][index] > real_losses[index]
                for index in range(len(records))
            ]
        ),
        "random_orthogonal_absolute_gap": exact.mean(
            [
                abs(
                    losses_by_variant["random_rotated"][index]
                    - losses_by_variant["orthogonal"][index]
                )
                for index in range(len(records))
            ]
        ),
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "records": len(records),
        "seeds": sorted({int(record["seed"]) for record in records}),
        "steps": [int(record["step"]) for record in records],
        "mean_no_step_oracle_loss": exact.mean(no_step),
        "mean_buffer_norm": exact.mean(
            [float(record["buffer_norm"]) for record in records]
        ),
        "mean_delta_norm": exact.mean(
            [float(record["delta_norm"]) for record in records]
        ),
        "mean_real_c_t": exact.mean([float(record["c_t"]) for record in records]),
        "mean_real_aligned_gain": exact.mean(
            [float(record["aligned_gain"]) for record in records]
        ),
        "max_buffer_checkpoint_relative_error": max(
            float(record["buffer_checkpoint_relative_error"]) for record in records
        ),
        "variants": per_variant,
        "orderings": orderings,
        "prediction": {
            "theory": (
                "d_t = A_t g_t + d_t_perp with A_t = 1 + mu + mu^2 c_t; at fixed "
                "|b| the aligned variant maximizes A_t and the anti-aligned "
                "variant minimizes it, so in the amplification-harmful regime "
                "one-step loss should order aligned > real > random_rotated ~ "
                "orthogonal > anti_aligned"
            ),
            "expected_loss_order_worst_to_best": [
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
        selected_indices = select_commit_groups(
            groups, args.commits, args.min_prior_rounds
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    selected = set(selected_indices)
    last_selected = max(selected)

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
        "variants": list(VARIANTS),
        "commits": args.commits,
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
    buffers: dict[int, torch.Tensor] = {}
    checkpoint_cache: tuple[str, object] | None = None
    records = []
    was_training = model.training
    model.eval()
    try:
        for group_index, group in enumerate(groups):
            if group_index > last_selected:
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
            merged_update = bn._production_merge_update(
                candidates, merge_momentum, frag
            )
            if not bool(torch.isfinite(merged_update).all()):
                raise SystemExit(
                    f"step={first['step']} fragment={fragment_id}: "
                    "merged group delta contains NaN or Inf"
                )
            delta = -merged_update

            if group_index in selected:
                if buffer_error > args.max_buffer_relative_error:
                    raise SystemExit(
                        f"step={first['step']} fragment={fragment_id}: reconstructed "
                        f"buffer diverged from the stored checkpoint momentum "
                        f"(relative_error={buffer_error:.3e} > "
                        f"{args.max_buffer_relative_error:.3e}); the capture likely "
                        "misses sync rounds for this fragment"
                    )
                records.append(
                    _intervene(
                        args,
                        model,
                        params,
                        layout,
                        oracle,
                        compute_loss,
                        checkpoint,
                        group_index,
                        group,
                        frag,
                        current,
                        buffer,
                        buffer_error,
                        merged_update,
                        delta,
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
                        f"[buffer-orientation] commits={len(records)}/"
                        f"{len(selected)} step={records[-1]['step']} "
                        f"fragment={fragment_id} "
                        f"buffer_rel_err={buffer_error:.2e}",
                        file=sys.stderr,
                        flush=True,
                    )

            buffers[fragment_id] = buffer.mul(args.outer_momentum).add(delta)
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
    group_index,
    group,
    frag,
    current,
    buffer,
    buffer_error,
    merged_update,
    delta,
    device,
) -> dict:
    first = group[0]
    fragment_id = int(first["fragment"])
    mu = args.outer_momentum
    try:
        geometry = buffer_geometry(buffer, delta, mu)
        generator = torch.Generator(device="cpu")
        commit_seed = variant_seed(args.seed, int(first["step"]), fragment_id)
        generator.manual_seed(commit_seed)
        variants = build_buffer_variants(buffer, delta, generator)
    except ValueError as exc:
        raise SystemExit(
            f"step={first['step']} fragment={fragment_id}: {exc}"
        ) from exc

    syncer_eval._apply_checkpoint(checkpoint, layout, params, device)
    oracle_current_loss, oracle_current_panel_losses = syncer_eval._losses(
        model, oracle["panels"], compute_loss
    )

    trials = {
        name: bn._nesterov_trial(
            current, variant, merged_update, args.outer_lr, mu
        )
        for name, variant in variants.items()
    }
    results = {}
    for name in VARIANTS:
        loss, panel_losses = exact.losses_for_trial(
            model,
            oracle["panels"],
            compute_loss,
            frag,
            params,
            current,
            trials[name],
            device,
        )
        results[name] = {"loss": loss, "panel_losses": panel_losses}

    real_panel_losses = results[BASELINE_VARIANT]["panel_losses"]
    variant_records = {}
    for name in VARIANTS:
        variant_geometry = buffer_geometry(variants[name], delta, mu)
        loss = results[name]["loss"]
        panel_losses = results[name]["panel_losses"]
        variant_records[name] = {
            "name": name,
            "buffer_norm": variant_geometry["buffer_norm"],
            "cosine_to_delta": variant_geometry["cosine_to_delta"],
            "c_t": variant_geometry["c_t"],
            "aligned_gain": variant_geometry["aligned_gain"],
            "transverse_ratio": variant_geometry["transverse_ratio"],
            "step_norm": float((trials[name] - current).norm().item()),
            "oracle_loss": loss,
            "oracle_panel_losses": panel_losses,
            "oracle_utility": oracle_current_loss - loss,
            "oracle_utility_se": exact.utility_se(
                oracle_current_panel_losses, panel_losses
            ),
            "paired_vs_no_step": paired_panel_stats(
                oracle_current_panel_losses, panel_losses
            ),
            "paired_vs_real": paired_panel_stats(real_panel_losses, panel_losses),
        }

    candidate_weights = [
        bn.soft._candidate_weight(row)
        for row in sorted(group, key=lambda item: int(item["learner_id"]))
    ]
    return {
        "schema": REPLAY_SCHEMA,
        "replay_config_sha256": args._replay_config_sha256,
        "seed": args.seed,
        "group_index": group_index,
        "group_ordinal": group_index + 1,
        "state_checkpoint": str(first["state_checkpoint"]),
        "step": int(first["step"]),
        "fragment": fragment_id,
        "merge_mode": "avg" if frag.merge_mode == MERGE_AVG else "rda",
        "outer_optimizer": "nesterov",
        "outer_lr": args.outer_lr,
        "outer_momentum": mu,
        "delta_correction": args.delta_correction,
        "candidate_count": len(group),
        "candidate_learner_ids": sorted(int(row["learner_id"]) for row in group),
        "candidate_weights": candidate_weights,
        "buffer_source": "recursion_replay",
        "buffer_checkpoint_relative_error": buffer_error,
        "buffer_norm": geometry["buffer_norm"],
        "delta_norm": geometry["delta_norm"],
        "buffer_cosine_to_delta": geometry["cosine_to_delta"],
        "c_t": geometry["c_t"],
        "r_t": geometry["r_t"],
        "aligned_gain": geometry["aligned_gain"],
        "transverse_ratio": geometry["transverse_ratio"],
        "variant_seed": commit_seed,
        "oracle_panels_sha256": oracle["metadata"]["panel_tensors_sha256"],
        "oracle_current_loss": oracle_current_loss,
        "oracle_current_panel_losses": oracle_current_panel_losses,
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
    parser.add_argument("--commits", type=int, default=10)
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
    if args.commits < 1:
        parser.error("--commits must be positive")
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
