#!/usr/bin/env python3
"""Train/test a lightweight calibrated fragment-utility score."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


CONTINUOUS_FEATURES = (
    "age",
    "freshness",
    "alignment",
    "uncertainty",
    "norm_anomaly",
    "update_norm_log",
    "c_steps",
    "c_tokens",
    "probe_grad_dot",
    "probe_grad_cosine",
    "probe_grad_normed_dot",
    "curvature_penalized_dot",
    "probe_grad_norm_log",
    "consensus_cosine",
    "consensus_normed_dot",
    "consensus_dot",
    "consensus_norm_log",
)


def _load_probe_summarizer():
    path = REPO_ROOT / "scripts" / "summarize_fragment_probe.py"
    spec = importlib.util.spec_from_file_location("_calibrated_probe_summary", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no probe records found")
    return rows


def split_rows(args, rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    if args.split == "heldout-learners":
        learners = sorted({int(r["learner_id"]) for r in rows})
        if args.test_learners:
            test_learners = {int(v) for v in args.test_learners.split(",") if v.strip()}
        else:
            k = max(1, math.ceil(len(learners) * args.test_fraction))
            test_learners = set(learners[-k:])
        train = [r for r in rows if int(r["learner_id"]) not in test_learners]
        test = [r for r in rows if int(r["learner_id"]) in test_learners]
        meta = {"test_learners": sorted(test_learners)}
    elif args.split == "late-rounds":
        ordered = sorted(rows, key=lambda r: (int(r.get("pull_step", 0)), int(r["learner_id"])))
        cut = max(1, min(len(ordered) - 1, round(len(ordered) * (1.0 - args.test_fraction))))
        train, test = ordered[:cut], ordered[cut:]
        meta = {"cut_index": cut}
    elif args.split == "heldout-seed":
        seeds = sorted({int(r["seed"]) for r in rows if "seed" in r})
        if not seeds:
            raise SystemExit("heldout-seed split requires records with a seed field")
        test_seed = int(args.test_seed) if args.test_seed is not None else seeds[-1]
        train = [r for r in rows if int(r["seed"]) != test_seed]
        test = [r for r in rows if int(r["seed"]) == test_seed]
        meta = {"test_seed": test_seed, "train_seeds": [s for s in seeds if s != test_seed]}
    else:
        ordered = list(rows)
        random.Random(args.seed).shuffle(ordered)
        cut = max(1, min(len(ordered) - 1, round(len(ordered) * (1.0 - args.test_fraction))))
        train, test = ordered[:cut], ordered[cut:]
        meta = {"seed": args.seed, "cut_index": cut}
    if not train or not test:
        raise SystemExit(f"split produced train={len(train)} test={len(test)}")
    return train, test, meta


def label_good(row: dict, strict: bool) -> float:
    if strict and row.get("utility_se") is not None:
        return 0.0 if float(row["utility"]) + float(row["utility_se"]) < 0.0 else 1.0
    return 1.0 if float(row["utility"]) > 0.0 else 0.0


def build_feature_spec(rows: list[dict], include_fragment: bool) -> dict:
    fragments = sorted({int(r["fragment"]) for r in rows}) if include_fragment else []
    return {"continuous": list(CONTINUOUS_FEATURES), "fragments": fragments}


def raw_continuous(row: dict, name: str) -> float:
    if name == "update_norm_log":
        return math.log1p(max(0.0, float(row.get("update_norm", 0.0))))
    if name == "probe_grad_norm_log":
        return math.log1p(max(0.0, float(row.get("probe_grad_norm", 0.0))))
    if name == "consensus_norm_log":
        return math.log1p(max(0.0, float(row.get("consensus_norm", 0.0))))
    return float(row.get(name, 0.0))


def fit_normalizer(rows: list[dict], spec: dict) -> tuple[list[float], list[float]]:
    means = []
    scales = []
    for name in spec["continuous"]:
        vals = [raw_continuous(r, name) for r in rows]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        scale = math.sqrt(var)
        means.append(mean)
        scales.append(scale if scale > 1e-8 else 1.0)
    return means, scales


def featurize(row: dict, spec: dict, means: list[float], scales: list[float]) -> list[float]:
    xs = [1.0]
    for i, name in enumerate(spec["continuous"]):
        xs.append((raw_continuous(row, name) - means[i]) / scales[i])
    fid = int(row["fragment"])
    for frag in spec["fragments"]:
        xs.append(1.0 if fid == frag else 0.0)
    return xs


def fit_logistic(
    train_rows: list[dict],
    *,
    spec: dict,
    means: list[float],
    scales: list[float],
    strict_label: bool,
    lr: float,
    l2: float,
    epochs: int,
) -> tuple[list[float], list[float]]:
    xs = [featurize(r, spec, means, scales) for r in train_rows]
    ys = [label_good(r, strict_label) for r in train_rows]
    weights = [0.0] * len(xs[0])
    losses = []
    n = len(xs)
    for _ in range(epochs):
        grad = [0.0] * len(weights)
        loss = 0.0
        for x, y in zip(xs, ys):
            z = sum(w * v for w, v in zip(weights, x))
            p = sigmoid(z)
            p = min(1.0 - 1e-8, max(1e-8, p))
            loss += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
            err = p - y
            for j, v in enumerate(x):
                grad[j] += err * v
        for j in range(1, len(weights)):
            loss += 0.5 * l2 * weights[j] * weights[j]
            grad[j] += l2 * weights[j]
        for j in range(len(weights)):
            weights[j] -= lr * grad[j] / n
        losses.append(loss / n)
    return weights, losses


def score_row(row: dict, spec: dict, means: list[float], scales: list[float], weights: list[float]) -> float:
    x = featurize(row, spec, means, scales)
    return sigmoid(sum(w * v for w, v in zip(weights, x)))


def annotate(rows: list[dict], spec: dict, means: list[float], scales: list[float], weights: list[float]) -> list[dict]:
    out = []
    for row in rows:
        r = dict(row)
        r["calibrated_score"] = score_row(row, spec, means, scales, weights)
        out.append(r)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument(
        "--split",
        choices=["heldout-learners", "late-rounds", "heldout-seed", "random"],
        default="heldout-learners",
    )
    p.add_argument("--test-learners", default=None)
    p.add_argument("--test-seed", default=None)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--strict-label", action="store_true", help="train on utility + utility_se < 0 when available")
    p.add_argument("--no-fragment-onehot", action="store_true")
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--l2", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    rows = read_rows(args.logs)
    train, test, split_meta = split_rows(args, rows)
    spec = build_feature_spec(train, include_fragment=not args.no_fragment_onehot)
    means, scales = fit_normalizer(train, spec)
    weights, losses = fit_logistic(
        train,
        spec=spec,
        means=means,
        scales=scales,
        strict_label=args.strict_label,
        lr=args.lr,
        l2=args.l2,
        epochs=args.epochs,
    )
    train_scored = annotate(train, spec, means, scales, weights)
    test_scored = annotate(test, spec, means, scales, weights)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "calibrated_train.jsonl"
    test_path = args.out_dir / "calibrated_test.jsonl"
    write_jsonl(train_path, train_scored)
    write_jsonl(test_path, test_scored)

    summarizer = _load_probe_summarizer()
    train_summary = summarizer.summarize([train_path])
    test_summary = summarizer.summarize([test_path])
    names = ["bias"] + spec["continuous"] + [f"fragment_{fid}" for fid in spec["fragments"]]
    summary = {
        "split": args.split,
        "split_meta": split_meta,
        "strict_label": args.strict_label,
        "train_records": len(train),
        "test_records": len(test),
        "features": names,
        "normalizer": {
            "means": dict(zip(spec["continuous"], means)),
            "scales": dict(zip(spec["continuous"], scales)),
        },
        "weights": dict(zip(names, weights)),
        "train_loss_initial": losses[0] if losses else None,
        "train_loss_final": losses[-1] if losses else None,
        "train_summary": summarizer.jsonable(train_summary),
        "test_summary": summarizer.jsonable(test_summary),
    }
    text = json.dumps(summarizer.jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    (args.out_dir / "calibration_summary.json").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
