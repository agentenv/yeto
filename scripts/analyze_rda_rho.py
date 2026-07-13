#!/usr/bin/env python
"""Per-tensor production-RDA autocorrelation and the two-term DiLoCo law.

EXP2.25 Correction follow-up. The original rho table in docs/EXP2_25.md was
computed on plain-average merged deltas over whole fragments; the corrected
two-term law (aligned overstep + buffer variance accumulation) requires the
per-tensor production-RDA autocorrelation as its rho input. This script:

1. Rebuilds each capture group's production merge exactly: per-tensor RDA
   (merge.rs `merge_rda`) with equal weights over the four learner candidates,
   anchored at the syncer's previous global fragment
   (Delta_m = Theta_p(prev) - theta_m,p). Anchors come from the capture's
   `state_before_step_%08d.ckpt` snapshots via S3 range reads; candidates are
   the locally mirrored flat f32 payloads. Correctness is verified end-to-end:
   for every mu=0 transition, Theta_prev - lr * RDA(...) must be bit-identical
   to the next same-fragment anchor.
2. Computes lag-1..4 autocorrelation of consecutive same-fragment RDA deltas
   per horizon, energy-weighted across tensors and pairs
   (rho = sum(dot) / sum(|a||b|)).
3. Fits the corrected two-term law against the 9-cell seed-223 loss grid
   (eta_eff = eta*(1 + mu/(1-mu*rho));
    A_RMS^2 = 1 + 2mu/(1-mu*rho) + mu^2(1+mu*rho)/((1-mu^2)(1-mu*rho)))
   and emits preregistered predictions for unseen cells.

Tensor layout: the capture index.jsonl carries no tensor boundaries and the
run's checkpoints predate the layout_meta trailer, so the layout is rebuilt
deterministically from the exported adapter (same `build_layout` binpack path
the learners used); the bit-exact step replay above proves the rebuild matches
production (fragment ids, tensor order, and boundaries).

Usage: python scripts/analyze_rda_rho.py [--workers 12] [--cache DIR]
Writes experiment-results/EXP2/rda-rho-law/{summary.json,summary.md}.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from yeto.fragments import build_layout  # noqa: E402

BUCKET = "yeto-exp-artifacts-533462777468-us-west-2"
RUN = "probecommit-resume-20260710"
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-shou-yeto/8fa1d046-c4da-41e1-a24e-0ae2ccff1020"
    "/scratchpad/autocorr"
)
NUM_FRAGMENTS = 4
NUMEL = 1352448  # per fragment, verified against checkpoint headers
CKPT_PARAMS_OFFSET = lambda frag: 16 + frag * (16 + 8 * NUMEL) + 16  # noqa: E731

# horizon -> (syncer_probe S3 prefix, outer lr of the capture arm)
# h64 candidates were verified by md5/ETag to be the EXP2.23 sync-SGD capture
# (vanilla-sync-seed223/sync-sgd028, eta=0.28, mu=0), the same capture E3 used.
HORIZONS = {
    16: (f"{RUN}/h-sweep-seed223/h16-mu0/work/m4/syncer_probe", 0.175),
    64: (f"{RUN}/vanilla-sync-seed223/sync-sgd028/work/m4/syncer_probe", 0.28),
    256: (f"{RUN}/h-sweep-seed223/h256-mu0/work/m4/syncer_probe", 0.175),
}
LOCAL_DIR = {16: SCRATCH / "h16", 64: SCRATCH / "h64", 256: SCRATCH / "h256"}
ADAPTER_KEY = f"{RUN}/h-sweep-seed223/h256-mu0/work/m4/export/adapter_model.safetensors"

MAX_LAG = 4
ETA_GRID = 0.175
ETA_STAR = 0.28  # tuned sync-SGD reference

# 9-cell seed-223 eval-loss grid (docs/EXP2_25.md E2 table).
LOSS_GRID = {
    (16, 0.0): 1.3519, (16, 0.5): 1.3632, (16, 0.9): 1.4383,
    (64, 0.0): 1.3578, (64, 0.5): 1.3614, (64, 0.9): 1.4193,
    (256, 0.0): 1.3805, (256, 0.5): 1.3674, (256, 0.9): 1.3977,
}

CAND_RE = re.compile(
    r"candidate_step_(\d{8})_fragment_(\d{4})_learner_(\d{4})\.f32$"
)


def s3_client():
    import boto3

    return boto3.client("s3")


def fetch_anchor(cache: Path, prefix: str, tag: str, step: int, frag: int) -> Path:
    """Download (or reuse) the params slice of fragment `frag` from the
    state-before-step checkpoint. Returns the local raw-f32 path."""
    out = cache / "anchors" / tag / f"step_{step:08d}_frag{frag}.f32"
    if out.exists() and out.stat().st_size == 4 * NUMEL:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    key = f"{prefix}/states/state_before_step_{step:08d}.ckpt"
    start = CKPT_PARAMS_OFFSET(frag)
    resp = s3_client().get_object(
        Bucket=BUCKET, Key=key, Range=f"bytes={start}-{start + 4 * NUMEL - 1}"
    )
    data = resp["Body"].read()
    if len(data) != 4 * NUMEL:
        raise RuntimeError(f"short read for {key}: {len(data)} bytes")
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(out)
    return out


def load_layout(cache: Path) -> list[list[tuple[str, int]]]:
    """Rebuild the production binpack layout from the exported adapter."""
    from safetensors import safe_open

    local = cache / "adapter_model.safetensors"
    if not local.exists():
        cache.mkdir(parents=True, exist_ok=True)
        s3_client().download_file(BUCKET, ADAPTER_KEY, str(local))
    f = safe_open(str(local), "np")
    named = []
    for k in f.keys():
        numel = int(np.prod(f.get_slice(k).get_shape()))
        # exported names lack peft's runtime ".default" adapter segment; the
        # insertion is uniform so lexicographic order (binpack tie-break) is
        # preserved, and the bit-exact replay check validates the result.
        fq = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        named.append((fq, numel))
    layout = build_layout(named, NUM_FRAGMENTS, "binpack")
    frags = [list(fr.tensors) for fr in layout.fragments]
    assert len(frags) == NUM_FRAGMENTS
    for fr in frags:
        assert sum(n for _, n in fr) == NUMEL
    return frags


def merge_rda_tensor(anchor: np.ndarray, cands: list[np.ndarray]) -> np.ndarray:
    """Equal-weight RDA over one tensor slice, mirroring merge.rs bit-for-bit:
    f32 deltas, f64 norm accumulation, f32 direction accumulation with
    coef = (0.25 / norm) cast to f32, degenerate fallback to direct average."""
    deltas = [anchor - c for c in cands]  # f32, matches rust (*a - *l)
    norms = [float(np.sqrt(np.sum(d.astype(np.float64) ** 2))) for d in deltas]
    radial = sum(norms) / len(cands)  # exact for equal power-of-two weights
    out = np.zeros(anchor.shape[0], dtype=np.float32)
    for d, n in zip(deltas, norms):
        if n == 0.0:
            continue
        out += np.float32(0.25 / n) * d
    mean_dir_norm = float(np.sqrt(np.sum(out.astype(np.float64) ** 2)))
    if mean_dir_norm < 1e-12:
        out = np.zeros(anchor.shape[0], dtype=np.float32)
        for d in deltas:
            out += np.float32(0.25) * d
        return out
    return out * np.float32(radial / mean_dir_norm)


def discover_groups(local_dir: Path) -> dict[int, dict[int, dict[int, Path]]]:
    """{step: {frag: {learner: path}}} from local candidate payloads."""
    groups: dict[int, dict[int, dict[int, Path]]] = defaultdict(lambda: defaultdict(dict))
    for p in sorted(local_dir.iterdir()):
        m = CAND_RE.search(p.name)
        if not m:
            continue
        step, frag, learner = (int(m.group(i)) for i in (1, 2, 3))
        groups[step][frag][learner] = p
    return groups


def analyze_horizon(h: int, cache: Path, frags: list[list[tuple[str, int]]],
                    workers: int) -> dict:
    prefix, lr = HORIZONS[h]
    tag = f"h{h}"
    groups = discover_groups(LOCAL_DIR[h])
    steps = sorted(groups)
    print(f"[h{h}] {len(steps)} capture steps, prefetching anchors...", flush=True)

    jobs = []
    for step in steps:
        (frag,) = groups[step].keys()
        jobs.append((step, frag))
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_anchor, cache, prefix, tag, s, f): (s, f) for s, f in jobs}
        done = 0
        for fut in cf.as_completed(futs):
            fut.result()
            done += 1
            if done % 40 == 0:
                print(f"[h{h}] anchors {done}/{len(jobs)}", flush=True)

    bounds = {}
    for fid, fr in enumerate(frags):
        b = np.cumsum([0] + [n for _, n in fr])
        bounds[fid] = b

    # per-fragment history of merged deltas: list of (step, delta_f32)
    history: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    prev_anchor: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    # accumulators
    lag_dot = {k: 0.0 for k in range(1, MAX_LAG + 1)}
    lag_nanb = {k: 0.0 for k in range(1, MAX_LAG + 1)}
    lag_pair_rhos = {k: [] for k in range(1, MAX_LAG + 1)}
    lag_frag_cos = {k: [] for k in range(1, MAX_LAG + 1)}
    # per-tensor lag-1 accumulators keyed by (frag, tensor_idx)
    t_dot = defaultdict(float)
    t_nanb = defaultdict(float)
    verify = {"exact": 0, "checked": 0, "max_abs_diff": 0.0}

    for step in steps:
        (frag,) = groups[step].keys()
        learners = groups[step][frag]
        assert sorted(learners) == [0, 1, 2, 3], (step, sorted(learners))
        anchor = np.fromfile(
            fetch_anchor(cache, prefix, tag, step, frag), dtype="<f4"
        )
        cands = [np.fromfile(learners[l], dtype="<f4") for l in range(4)]
        b = bounds[frag]
        merged = np.empty(NUMEL, dtype=np.float32)
        for i in range(len(b) - 1):
            merged[b[i]:b[i + 1]] = merge_rda_tensor(
                anchor[b[i]:b[i + 1]], [c[b[i]:b[i + 1]] for c in cands]
            )

        # end-to-end verification: previous same-fragment anchor + its merged
        # delta must reproduce this anchor bit-exactly (mu=0, multiplier x1).
        if frag in prev_anchor:
            pstep, panchor, pmerged = prev_anchor[frag]
            pred = panchor - np.float32(lr) * pmerged
            diff = float(np.max(np.abs(pred - anchor)))
            verify["checked"] += 1
            verify["max_abs_diff"] = max(verify["max_abs_diff"], diff)
            if diff == 0.0:
                verify["exact"] += 1
        prev_anchor[frag] = (step, anchor, merged)

        # autocorrelation vs the last MAX_LAG same-fragment deltas
        hist = history[frag]
        for k in range(1, MAX_LAG + 1):
            if len(hist) < k:
                break
            _, prev = hist[-k]
            pair_dot = 0.0
            pair_nanb = 0.0
            for i in range(len(b) - 1):
                a = merged[b[i]:b[i + 1]].astype(np.float64)
                pv = prev[b[i]:b[i + 1]].astype(np.float64)
                d = float(np.dot(a, pv))
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(pv))
                pair_dot += d
                pair_nanb += na * nb
                if k == 1:
                    t_dot[(frag, i)] += d
                    t_nanb[(frag, i)] += na * nb
            lag_dot[k] += pair_dot
            lag_nanb[k] += pair_nanb
            lag_pair_rhos[k].append(pair_dot / pair_nanb)
            af = merged.astype(np.float64)
            pf = prev.astype(np.float64)
            lag_frag_cos[k].append(
                float(np.dot(af, pf) / (np.linalg.norm(af) * np.linalg.norm(pf)))
            )
        hist.append((step, merged))
        if len(hist) > MAX_LAG:
            hist.pop(0)

    tensor_rhos = np.array(
        [t_dot[key] / t_nanb[key] for key in sorted(t_dot)] or [np.nan]
    )
    res = {
        "steps": len(steps),
        "capture_prefix": f"s3://{BUCKET}/{prefix}",
        "capture_lr": lr,
        "verification": verify,
        "lags": {},
        "tensor_lag1": {
            "n_tensors": int(tensor_rhos.size),
            "mean": float(np.mean(tensor_rhos)),
            "p10": float(np.percentile(tensor_rhos, 10)),
            "p50": float(np.percentile(tensor_rhos, 50)),
            "p90": float(np.percentile(tensor_rhos, 90)),
            "min": float(np.min(tensor_rhos)),
            "max": float(np.max(tensor_rhos)),
        },
    }
    for k in range(1, MAX_LAG + 1):
        pr = np.array(lag_pair_rhos[k]) if lag_pair_rhos[k] else np.array([np.nan])
        res["lags"][k] = {
            "pairs": len(lag_pair_rhos[k]),
            "rho_energy_weighted": lag_dot[k] / lag_nanb[k] if lag_nanb[k] else None,
            "pair_rho_mean": float(np.mean(pr)),
            "pair_rho_p10": float(np.percentile(pr, 10)),
            "pair_rho_p90": float(np.percentile(pr, 90)),
            "fragment_cos_mean": float(np.mean(lag_frag_cos[k]))
            if lag_frag_cos[k] else None,
        }
    return res


def eta_eff(eta: float, mu: float, rho: float) -> float:
    return eta * (1.0 + mu / (1.0 - mu * rho)) if mu > 0 else eta


def a2_rms(mu: float, rho: float) -> float:
    if mu == 0:
        return 1.0
    return (
        1.0
        + 2.0 * mu / (1.0 - mu * rho)
        + mu * mu * (1.0 + mu * rho) / ((1.0 - mu * mu) * (1.0 - mu * rho))
    )


def fit_law(rho_by_h: dict[int, float]) -> dict:
    hs = [16, 64, 256]
    cells = sorted(LOSS_GRID)
    y = np.array([LOSS_GRID[c] for c in cells])

    def features(h, mu, eta):
        rho = rho_by_h[h]
        e = eta_eff(eta, mu, rho)
        return math.log(e / ETA_STAR) ** 2, math.log(a2_rms(mu, rho))

    # design: per-H intercepts + b * log(eta_eff/eta*)^2 + v * log(A2_RMS)
    X = np.zeros((len(cells), len(hs) + 2))
    for r, (h, mu) in enumerate(cells):
        X[r, hs.index(h)] = 1.0
        x1, x2 = features(h, mu, ETA_GRID)
        X[r, 3], X[r, 4] = x1, x2
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    # comparison model A: aligned term only (corrected eta_eff, no variance)
    Xa = X[:, :4]
    ca, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    ra = y - Xa @ ca
    # comparison model B: superseded single-term law eta_eff = eta/(1-mu*rho)
    Xb = np.zeros((len(cells), len(hs) + 1))
    for r, (h, mu) in enumerate(cells):
        Xb[r, hs.index(h)] = 1.0
        e_old = ETA_GRID / (1.0 - mu * rho_by_h[h])
        Xb[r, 3] = math.log(e_old / ETA_STAR) ** 2
    cb, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    rb = y - Xb @ cb

    return {
        "model": "loss = c_H + b*log(eta_eff/0.28)^2 + v*log(A2_RMS)",
        "eta_grid": ETA_GRID,
        "eta_star": ETA_STAR,
        "intercepts": {str(h): float(coef[i]) for i, h in enumerate(hs)},
        "b_aligned": float(coef[3]),
        "v_variance": float(coef[4]),
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": math.sqrt(ss_res / len(cells)),
        "max_abs_resid": float(np.max(np.abs(resid))),
        "cells": [
            {
                "H": h, "mu": mu, "loss": LOSS_GRID[(h, mu)],
                "eta_eff": eta_eff(ETA_GRID, mu, rho_by_h[h]),
                "A2_RMS": a2_rms(mu, rho_by_h[h]),
                "pred": float(pred[r]), "resid": float(resid[r]),
            }
            for r, (h, mu) in enumerate(cells)
        ],
        "comparison": {
            "aligned_only_corrected": {
                "rmse": float(np.sqrt(np.mean(ra**2))),
                "r2": 1.0 - float(np.sum(ra**2)) / ss_tot,
            },
            "superseded_single_term": {
                "rmse": float(np.sqrt(np.mean(rb**2))),
                "r2": 1.0 - float(np.sum(rb**2)) / ss_tot,
            },
        },
        "_coef": coef.tolist(),
    }


def predictions(fit: dict, rho_by_h: dict[int, float]) -> dict:
    hs = [16, 64, 256]
    coef = np.array(fit["_coef"])

    def predict(h, mu, eta):
        rho = rho_by_h[h]
        x = np.zeros(5)
        x[hs.index(h)] = 1.0
        x[3] = math.log(eta_eff(eta, mu, rho) / ETA_STAR) ** 2
        x[4] = math.log(a2_rms(mu, rho))
        return float(x @ coef)

    p16 = predict(16, 0.9, ETA_GRID)
    p256 = predict(256, 0.9, ETA_GRID)
    p64_e28 = predict(64, 0.5, 0.28)
    return {
        "preregistered": True,
        "note": (
            "seed-251 predictions reuse seed-223 per-H intercepts and rho; the "
            "preregistered claims are the orderings and gaps, not absolute levels"
        ),
        "seed251_h16_mu09_pred_loss": p16,
        "seed251_h256_mu09_pred_loss": p256,
        "seed251_mu09_ordering": "loss(H=16,mu=0.9) > loss(H=256,mu=0.9)",
        "seed251_mu09_gap": p16 - p256,
        "seed251_h16_mu09_penalty_vs_h16_mu0": p16 - predict(16, 0.0, ETA_GRID),
        "seed251_h256_mu09_penalty_vs_h256_mu05": p256 - predict(256, 0.5, ETA_GRID),
        "h64_mu05_eta028": {
            "eta_eff": eta_eff(0.28, 0.5, rho_by_h[64]),
            "A2_RMS": a2_rms(0.5, rho_by_h[64]),
            "pred_loss": p64_e28,
            "vs_h64_mu05_eta0175": p64_e28 - predict(64, 0.5, ETA_GRID),
            "vs_h64_mu0_eta0175": p64_e28 - predict(64, 0.0, ETA_GRID),
        },
    }


def write_summary_md(out: Path, results: dict) -> None:
    r = results
    lines = [
        "# Per-tensor RDA autocorrelation and the two-term law (EXP2.25 Correction follow-up)",
        "",
        "rho is computed on production-RDA merged deltas (per-tensor `merge_rda`, equal",
        "weights, anchored at the previous global fragment), energy-weighted across",
        "tensors and pairs: rho = sum(dot) / sum(|a||b|). Verification: for every mu=0",
        "transition the replayed outer step (Theta_prev - lr * RDA delta) was compared",
        "bit-for-bit against the next same-fragment anchor checkpoint.",
        "",
        "## RDA rho by horizon",
        "",
        "| H | pairs (lag1) | rho lag1 | lag2 | lag3 | lag4 | tensor p10 | tensor p90 | replay exact |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in (16, 64, 256):
        hr = r["horizons"][str(h)]
        lg = hr["lags"]
        v = hr["verification"]
        lines.append(
            f"| {h} | {lg['1']['pairs']} | {lg['1']['rho_energy_weighted']:.4f} "
            f"| {lg['2']['rho_energy_weighted']:.4f} | {lg['3']['rho_energy_weighted']:.4f} "
            f"| {lg['4']['rho_energy_weighted']:.4f} | {hr['tensor_lag1']['p10']:.3f} "
            f"| {hr['tensor_lag1']['p90']:.3f} | {v['exact']}/{v['checked']} |"
        )
    fit = r["fit"]
    lines += [
        "",
        "## Two-term law fit (9-cell seed-223 grid)",
        "",
        f"Model: `{fit['model']}` with per-H intercepts.",
        f"- b (aligned overstep) = {fit['b_aligned']:.4f}",
        f"- v (variance accumulation) = {fit['v_variance']:.4f}",
        f"- R^2 = {fit['r2']:.4f}, RMSE = {fit['rmse']:.5f}, max |resid| = {fit['max_abs_resid']:.5f}",
        f"- aligned-only (corrected eta_eff): RMSE {fit['comparison']['aligned_only_corrected']['rmse']:.5f}"
        f" (R^2 {fit['comparison']['aligned_only_corrected']['r2']:.3f})",
        f"- superseded single-term eta/(1-mu*rho): RMSE {fit['comparison']['superseded_single_term']['rmse']:.5f}"
        f" (R^2 {fit['comparison']['superseded_single_term']['r2']:.3f})",
        "",
        "| H | mu | eta_eff | A2_RMS | loss | pred | resid |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in fit["cells"]:
        lines.append(
            f"| {c['H']} | {c['mu']} | {c['eta_eff']:.4f} | {c['A2_RMS']:.2f} "
            f"| {c['loss']:.4f} | {c['pred']:.4f} | {c['resid']:+.4f} |"
        )
    p = r["predictions"]
    lines += [
        "",
        "## Preregistered predictions (unseen cells)",
        "",
        f"1. seed-251, H=16 mu=0.9: predicted loss {p['seed251_h16_mu09_pred_loss']:.4f}",
        f"2. seed-251, H=256 mu=0.9: predicted loss {p['seed251_h256_mu09_pred_loss']:.4f}",
        f"3. Ordering: {p['seed251_mu09_ordering']} by {p['seed251_mu09_gap']:.4f}",
        f"   (H=16 mu=0.9 penalty vs its mu=0 sibling {p['seed251_h16_mu09_penalty_vs_h16_mu0']:+.4f};"
        f" H=256 mu=0.9 vs mu=0.5 sibling {p['seed251_h256_mu09_penalty_vs_h256_mu05']:+.4f})",
        f"4. H=64 mu=0.5 eta=0.28: eta_eff {p['h64_mu05_eta028']['eta_eff']:.4f},"
        f" predicted loss {p['h64_mu05_eta028']['pred_loss']:.4f}"
        f" ({p['h64_mu05_eta028']['vs_h64_mu05_eta0175']:+.4f} vs the eta=0.175 mu=0.5 cell,"
        f" {p['h64_mu05_eta028']['vs_h64_mu0_eta0175']:+.4f} vs the eta=0.175 mu=0 cell)",
        "",
        "## Caveats",
        "",
        "- rho is measured on mu=0 captures (open loop); closed-loop rho under momentum",
        "  differs. H=64 rho comes from the EXP2.23 sync-SGD capture at eta=0.28",
        "  (verified by content hash), while H=16/H=256 arms ran at eta=0.175.",
        "- seed-251 predictions reuse seed-223 intercepts and rho; absolute levels may",
        "  shift with seed, the preregistered content is the orderings and gaps.",
        "- LoRA rank 2 may structurally inflate rho (see EXP2_25.md).",
    ]
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/rda_states/cache"))
    ap.add_argument(
        "--out", type=Path,
        default=REPO / "experiment-results/EXP2/rda-rho-law",
    )
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    frags = load_layout(args.cache)
    results = {"horizons": {}}
    for h in (16, 64, 256):
        results["horizons"][str(h)] = analyze_horizon(h, args.cache, frags, args.workers)
        v = results["horizons"][str(h)]["verification"]
        print(
            f"[h{h}] rho_lag1={results['horizons'][str(h)]['lags'][1]['rho_energy_weighted']:.4f} "
            f"replay exact {v['exact']}/{v['checked']} (max diff {v['max_abs_diff']:.3g})",
            flush=True,
        )

    rho_by_h = {
        h: results["horizons"][str(h)]["lags"][1]["rho_energy_weighted"]
        for h in (16, 64, 256)
    }
    fit = fit_law(rho_by_h)
    results["rho_lag1_energy_weighted"] = {str(h): rho_by_h[h] for h in rho_by_h}
    results["fit"] = fit
    results["predictions"] = predictions(fit, rho_by_h)
    results["loss_grid"] = {f"H{h}_mu{mu}": v for (h, mu), v in sorted(LOSS_GRID.items())}

    # JSON-serializable lag keys; keep _coef in JSON for reproducibility
    for hr in results["horizons"].values():
        hr["lags"] = {str(k): v for k, v in hr["lags"].items()}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    write_summary_md(args.out / "summary.md", results)
    print(f"wrote {args.out}/summary.json and summary.md")


if __name__ == "__main__":
    main()
