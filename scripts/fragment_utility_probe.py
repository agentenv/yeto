#!/usr/bin/env python3
"""Balanced asynchronous fragment-utility probe.

This is a lightweight local experiment for testing whether coarse accounting
signals predict the short-horizon utility of asynchronous fragment updates.
It uses a balanced stochastic quadratic objective with cross-fragment coupling:
all learners draw from the same data distribution, but responses arrive with
different ages. The probe records exact one-step utility for each returned
fragment and compares token count, freshness, alignment, anomaly, and a
combined score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Candidate:
    token_count: float
    freshness: float
    alignment: float
    uncertainty: float
    norm_anomaly: float
    sensitivity: float
    combined_score: float
    utility: float
    learner: int
    fragment: int
    age: int
    norm: float


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC where label 1 should receive a larger score."""
    labels = labels.astype(bool)
    pos = int(labels.sum())
    neg = int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    pos_rank_sum = float(ranks[labels].sum())
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def expected_calibration_error(labels_good: np.ndarray, prob_good: np.ndarray, bins: int = 10) -> float:
    labels_good = labels_good.astype(np.float64)
    prob_good = np.clip(prob_good.astype(np.float64), 0.0, 1.0)
    ece = 0.0
    for lo in np.linspace(0.0, 1.0, bins, endpoint=False):
        hi = lo + 1.0 / bins
        if hi >= 1.0:
            mask = (prob_good >= lo) & (prob_good <= hi)
        else:
            mask = (prob_good >= lo) & (prob_good < hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(prob_good[mask].mean()) - float(labels_good[mask].mean()))
    return ece


def zscore_to_probability(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    if std < 1e-12:
        return np.full_like(x, 0.5, dtype=np.float64)
    z = (x - float(x.mean())) / std
    return 1.0 / (1.0 + np.exp(-z))


def make_spd_problem(rng: np.random.Generator, dim: int, coupling_rank: int, coupling: float):
    diag = rng.lognormal(mean=-0.25, sigma=0.55, size=dim).astype(np.float64)
    low_rank = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(dim, coupling_rank))
    a = np.diag(diag) + coupling * (low_rank @ low_rank.T)
    # Keep the local optimizer stable across seeds.
    lam_max = float(np.linalg.eigvalsh(a).max())
    a /= lam_max
    x_star = rng.normal(0.0, 0.6, size=dim)
    theta = x_star + rng.normal(0.0, 1.0, size=dim)
    return a, x_star, theta, np.diag(a).copy()


def loss(a: np.ndarray, x_star: np.ndarray, theta: np.ndarray) -> float:
    d = theta - x_star
    return 0.5 * float(d @ a @ d)


def local_train(
    rng: np.random.Generator,
    a: np.ndarray,
    x_star: np.ndarray,
    start: np.ndarray,
    *,
    steps: int,
    lr: float,
    noise: float,
    momentum: float,
) -> np.ndarray:
    x = start.copy()
    v = np.zeros_like(x)
    for _ in range(steps):
        grad = a @ (x - x_star)
        grad += rng.normal(0.0, noise, size=x.shape)
        v = momentum * v + grad
        x -= lr * v
    return x


def sample_age(rng: np.random.Generator, learner: int) -> int:
    group = learner % 3
    if group == 0:  # fast responses
        return int(min(5, rng.poisson(0.8)))
    if group == 1:  # medium responses
        return int(min(12, 1 + rng.poisson(2.7)))
    # delayed responses
    return int(min(28, 4 + rng.negative_binomial(3, 0.38)))


def simulate(args: argparse.Namespace) -> tuple[list[Candidate], dict]:
    rng = np.random.default_rng(args.seed)
    a, x_star, theta, curvature_diag = make_spd_problem(
        rng, args.dim, args.coupling_rank, args.coupling
    )
    if args.dim % args.fragments != 0:
        raise SystemExit("--dim must be divisible by --fragments")
    frag_dim = args.dim // args.fragments
    slices = [slice(i * frag_dim, (i + 1) * frag_dim) for i in range(args.fragments)]
    sensitivity_raw = np.array([float(curvature_diag[s].mean()) for s in slices])
    sensitivity = (sensitivity_raw - sensitivity_raw.mean()) / max(float(sensitivity_raw.std()), 1e-12)

    history: deque[np.ndarray] = deque([theta.copy()], maxlen=args.history)
    momentum = [np.zeros(frag_dim, dtype=np.float64) for _ in range(args.fragments)]
    norm_history: list[deque[float]] = [deque(maxlen=96) for _ in range(args.fragments)]
    learner_norm_ema = defaultdict(float)
    learner_norm_var = defaultdict(lambda: 1e-4)
    candidates: list[Candidate] = []
    merge_utilities: list[float] = []
    token_mass_bad: list[float] = []
    token_cv_by_round: list[float] = []

    for t in range(args.rounds):
        fragment = t % args.fragments
        s = slices[fragment]
        current_loss = loss(a, x_star, theta)
        grad_fragment = a[s, :] @ (theta - x_star)
        if np.linalg.norm(momentum[fragment]) < 1e-12:
            align_ref = -grad_fragment
        else:
            align_ref = momentum[fragment]

        round_candidates: list[tuple[Candidate, np.ndarray]] = []
        for learner in range(args.learners):
            age = min(sample_age(rng, learner), len(history) - 1)
            base = history[-1 - age]
            local_seed = np.random.default_rng(args.seed * 1_000_003 + t * 997 + learner)
            local = local_train(
                local_seed,
                a,
                x_star,
                base,
                steps=args.local_steps,
                lr=args.inner_lr,
                noise=args.grad_noise,
                momentum=args.inner_momentum,
            )
            update = local[s] - theta[s]
            burst_prob = args.burst_prob * (1.0 + min(age, 12) / 12.0)
            if rng.random() < burst_prob:
                ref_norm = np.linalg.norm(align_ref)
                update_norm = np.linalg.norm(update)
                if ref_norm > 1e-12 and update_norm > 1e-12:
                    # Rare optimizer-state bursts: still directionally
                    # plausible, but too large for the current fragment.
                    unit = align_ref / ref_norm
                    update = update + (
                        rng.lognormal(mean=math.log(args.burst_scale), sigma=0.25)
                        * update_norm
                        * unit
                    )
            token_count = args.local_steps * args.seq_len * float(
                np.clip(rng.normal(1.0, args.token_jitter), 0.85, 1.15)
            )
            trial = theta.copy()
            trial[s] += args.outer_lr * update
            utility = current_loss - loss(a, x_star, trial)
            freshness = math.exp(-age / args.freshness_scale)
            alignment = cosine(update, align_ref)
            norm = float(np.linalg.norm(update))

            hist = norm_history[fragment]
            if hist:
                med = float(np.median(hist))
                mad = float(np.median(np.abs(np.array(hist) - med))) + 1e-8
                norm_anomaly = abs(norm - med) / mad
            else:
                norm_anomaly = 0.0

            key = (learner, fragment)
            prev_mean = learner_norm_ema[key]
            prev_var = learner_norm_var[key]
            if prev_mean == 0.0:
                learner_norm_ema[key] = norm
                uncertainty = 0.0
            else:
                delta = norm - prev_mean
                learner_norm_ema[key] = 0.85 * prev_mean + 0.15 * norm
                learner_norm_var[key] = 0.85 * prev_var + 0.15 * delta * delta
                uncertainty = math.sqrt(learner_norm_var[key]) / (abs(learner_norm_ema[key]) + 1e-8)

            combined_logit = (
                2.25 * alignment
                + 1.35 * freshness
                - 0.55 * math.log1p(norm_anomaly)
                - 0.80 * uncertainty
                + 0.25 * sensitivity[fragment]
            )
            combined_score = sigmoid(combined_logit)
            cand = Candidate(
                token_count=token_count,
                freshness=freshness,
                alignment=alignment,
                uncertainty=uncertainty,
                norm_anomaly=norm_anomaly,
                sensitivity=float(sensitivity[fragment]),
                combined_score=combined_score,
                utility=utility,
                learner=learner,
                fragment=fragment,
                age=age,
                norm=norm,
            )
            round_candidates.append((cand, update))
            candidates.append(cand)

        tokens = np.array([c.token_count for c, _ in round_candidates], dtype=np.float64)
        token_cv_by_round.append(float(tokens.std() / tokens.mean()))
        weights = tokens * tokens / args.local_steps
        weights /= weights.sum()
        merged_update = np.zeros(frag_dim, dtype=np.float64)
        for weight, (_, update) in zip(weights, round_candidates):
            merged_update += weight * update
        trial = theta.copy()
        trial[s] += args.outer_lr * merged_update
        merge_utilities.append(current_loss - loss(a, x_star, trial))
        bad_mass = float(sum(w for w, (c, _) in zip(weights, round_candidates) if c.utility < 0.0))
        token_mass_bad.append(bad_mass)

        theta[s] += args.outer_lr * merged_update
        momentum[fragment] = args.outer_momentum * momentum[fragment] + merged_update
        for _, update in round_candidates:
            norm_history[fragment].append(float(np.linalg.norm(update)))
        history.append(theta.copy())

    diagnostics = {
        "final_loss": loss(a, x_star, theta),
        "negative_merge_rate": float(np.mean(np.array(merge_utilities) < 0.0)),
        "bad_token_mass_mean": float(np.mean(token_mass_bad)),
        "bad_token_mass_p95": float(np.quantile(token_mass_bad, 0.95)),
        "token_cv_mean": float(np.mean(token_cv_by_round)),
        "token_cv_p95": float(np.quantile(token_cv_by_round, 0.95)),
    }
    return candidates, diagnostics


def summarize(candidates: list[Candidate], diagnostics: dict) -> dict:
    rows = {
        "token_count": np.array([c.token_count for c in candidates]),
        "freshness": np.array([c.freshness for c in candidates]),
        "alignment": np.array([c.alignment for c in candidates]),
        "norm_anomaly": np.array([c.norm_anomaly for c in candidates]),
        "combined_score": np.array([c.combined_score for c in candidates]),
        "utility": np.array([c.utility for c in candidates]),
    }
    utility = rows["utility"]
    good = utility > 0.0
    bad = ~good
    table = {}
    orientations = {
        "token_count": rows["token_count"],
        "freshness": rows["freshness"],
        "alignment": rows["alignment"],
        "norm_anomaly": -rows["norm_anomaly"],
        "combined_score": rows["combined_score"],
    }
    for name, signal in orientations.items():
        prob_good = zscore_to_probability(signal)
        if name == "combined_score":
            prob_good = rows["combined_score"]
        table[name] = {
            "pearson_utility": pearson(signal, utility),
            "spearman_utility": spearman(signal, utility),
            "bad_fragment_auroc": auroc(bad, -signal),
            "calibration_error": expected_calibration_error(good, prob_good),
        }

    summary = {
        "n_candidates": len(candidates),
        "negative_utility_rate": float(np.mean(bad)),
        "utility_quantiles": {
            "p05": float(np.quantile(utility, 0.05)),
            "p50": float(np.quantile(utility, 0.50)),
            "p95": float(np.quantile(utility, 0.95)),
        },
        "diagnostics": diagnostics,
        "signals": table,
    }
    return summary


def write_candidates(path: Path, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "learner",
                "fragment",
                "age",
                "token_count",
                "freshness",
                "alignment",
                "uncertainty",
                "norm_anomaly",
                "sensitivity",
                "combined_score",
                "utility",
                "norm",
            ],
        )
        writer.writeheader()
        for c in candidates:
            writer.writerow(c.__dict__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--learners", type=int, default=12)
    p.add_argument("--fragments", type=int, default=12)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--rounds", type=int, default=600)
    p.add_argument("--history", type=int, default=64)
    p.add_argument("--local-steps", type=int, default=36)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--inner-lr", type=float, default=0.18)
    p.add_argument("--inner-momentum", type=float, default=0.45)
    p.add_argument("--outer-lr", type=float, default=0.55)
    p.add_argument("--outer-momentum", type=float, default=0.85)
    p.add_argument("--grad-noise", type=float, default=0.012)
    p.add_argument("--token-jitter", type=float, default=0.035)
    p.add_argument("--burst-prob", type=float, default=0.055)
    p.add_argument("--burst-scale", type=float, default=3.8)
    p.add_argument("--freshness-scale", type=float, default=7.0)
    p.add_argument("--coupling-rank", type=int, default=32)
    p.add_argument("--coupling", type=float, default=3.8)
    p.add_argument("--out-dir", type=Path, default=Path("experiment-work/fragment_utility_probe"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    candidates, diagnostics = simulate(args)
    summary = summarize(candidates, diagnostics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_candidates(args.out_dir / "candidates.csv", candidates)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
