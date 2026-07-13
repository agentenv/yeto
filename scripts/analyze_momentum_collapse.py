#!/usr/bin/env python
"""EXP2.39 momentum-penalty collapse test (executes the frozen prereg).

Given the geometry panel (per-cell Z means, scripts/analyze_geometry_panel.py)
and a cells file pairing each mu>0 arm with its mu=0 partner, fit each candidate
collapse variable Z to the momentum penalty Delta = loss(mu>0) - loss(mu=0) on
the pre-registered TRAIN cells, freeze the OLS (slope, intercept), score the
HELD-OUT cells, and evaluate the pre-registered acceptance criteria (A-D in
exp2_39_threshold_prereg.md). No Z or split is chosen here; they are inputs.

Inputs
  --panel panel.json   list of run summaries; each has `label`, `mean_gnorm`,
                       `eta`, and (when a capture existed) capture.means with
                       rho/c_t/r_t/agree_mean/lambda_hat/r2_lambda.
  --cells cells.json   list of {cell, z_label, mu0_loss, mu_loss, split}
                       where z_label indexes the mu>0 arm in the panel, split is
                       "train" or "heldout". mu0_loss/mu_loss are the paired
                       held-out CE losses (results.jsonl eval_loss / refs).

Z definitions (from the panel means of the mu>0 arm; eta from the panel):
  r_t, r2_lambda, eta2_r2_lambda = eta^2 * r2_lambda, rho, c_t, lambda_hat,
  disagree = 1 - agree_mean, mean_gnorm (the norm null).

Writes collapse.json / collapse.md to --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NOISE_FLOOR = 0.009


def z_values(summary: dict) -> dict[str, float | None]:
    means = (summary.get("capture") or {}).get("means", {})
    eta = float(summary.get("eta", 0.0))
    r2l = means.get("r2_lambda")
    out = {
        "r_t": means.get("r_t"),
        "r2_lambda": r2l,
        "eta2_r2_lambda": (eta * eta * r2l) if r2l is not None else None,
        "rho": means.get("rho"),
        "c_t": means.get("c_t"),
        "lambda_hat": means.get("lambda_hat"),
        "disagree": (1.0 - means["agree_mean"]) if "agree_mean" in means else None,
        "mean_gnorm": summary.get("mean_gnorm"),
    }
    return out


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """slope, intercept via least squares y ~ a x + b."""
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def r2_against_baseline(y: np.ndarray, yhat: np.ndarray, baseline: float) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - baseline) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument(
        "--out", type=Path, default=Path("experiment-results/EXP2/geometry-panel")
    )
    args = ap.parse_args()

    panel = {s["label"]: s for s in json.loads(args.panel.read_text())}
    cells = json.loads(args.cells.read_text())

    # attach Z vector + penalty to each cell
    rows = []
    for c in cells:
        s = panel.get(c["z_label"])
        if s is None:
            raise SystemExit(f"cell {c['cell']}: z_label {c['z_label']} not in panel")
        zs = z_values(s)
        rows.append(
            {
                "cell": c["cell"],
                "split": c["split"],
                "delta": float(c["mu_loss"]) - float(c["mu0_loss"]),
                "z": zs,
            }
        )

    z_names = list(rows[0]["z"].keys())
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] == "heldout"]
    train_mean_delta = float(np.mean([r["delta"] for r in train]))

    results = {}
    for zn in z_names:
        tr = [(r["z"][zn], r["delta"]) for r in train if r["z"][zn] is not None]
        hd = [(r["z"][zn], r["delta"], r["cell"]) for r in held if r["z"][zn] is not None]
        if len(tr) < 3:
            results[zn] = {"skipped": f"only {len(tr)} train cells with Z"}
            continue
        xtr = np.array([t[0] for t in tr])
        ytr = np.array([t[1] for t in tr])
        a, b = ols(xtr, ytr)
        r2_train = r2_against_baseline(ytr, a * xtr + b, float(np.mean(ytr)))
        entry = {
            "slope": a,
            "intercept": b,
            "r2_train": r2_train,
            "n_train": len(tr),
            "tau": (NOISE_FLOOR - b) / a if a != 0 else None,
        }
        if hd:
            xhd = np.array([h[0] for h in hd])
            yhd = np.array([h[1] for h in hd])
            yhat = a * xhd + b
            entry["r2_heldout"] = r2_against_baseline(yhd, yhat, train_mean_delta)
            entry["n_heldout"] = len(hd)
            entry["heldout_pred"] = [
                {"cell": h[2], "z": float(h[0]), "delta": float(h[1]),
                 "pred": float(a * h[0] + b)}
                for h in hd
            ]
        results[zn] = entry

    # acceptance for the pre-registered primary (Z2 = r2_lambda) with Z3 fallback
    null = results.get("mean_gnorm", {})
    verdict = {}
    for primary in ("r2_lambda", "eta2_r2_lambda"):
        e = results.get(primary, {})
        if "r2_train" not in e:
            continue
        A = e["r2_train"] >= 0.80
        B = e.get("r2_heldout", -9) >= 0.60
        C = e.get("r2_heldout", -9) - null.get("r2_heldout", -9) >= 0.20
        verdict[primary] = {
            "A_train_r2>=0.80": A,
            "B_heldout_r2>=0.60": B,
            "C_beats_norm_null_by>=0.20": C,
            "r2_train": e["r2_train"],
            "r2_heldout": e.get("r2_heldout"),
            "norm_null_r2_heldout": null.get("r2_heldout"),
            "passes_ABC": bool(A and B and C),
        }
    # best Z by held-out R^2
    ranked = sorted(
        ((zn, e.get("r2_heldout")) for zn, e in results.items()
         if isinstance(e, dict) and e.get("r2_heldout") is not None),
        key=lambda t: t[1], reverse=True,
    )

    out = {
        "noise_floor": NOISE_FLOOR,
        "train_cells": [r["cell"] for r in train],
        "heldout_cells": [r["cell"] for r in held],
        "per_Z": results,
        "primary_verdict": verdict,
        "ranked_by_heldout_r2": ranked,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "collapse.json").write_text(json.dumps(out, indent=2) + "\n")

    lines = [
        "# EXP2.39 momentum-penalty collapse (frozen prereg)",
        "",
        f"train cells ({len(train)}): " + ", ".join(r["cell"] for r in train),
        f"held-out cells ({len(held)}): " + ", ".join(r["cell"] for r in held),
        "",
        "| Z | slope | intercept | R2_train | R2_heldout | tau |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for zn in z_names:
        e = results[zn]
        if "r2_train" not in e:
            lines.append(f"| {zn} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {zn} | {e['slope']:.4g} | {e['intercept']:.4g} | "
            f"{e['r2_train']:.3f} | "
            f"{e.get('r2_heldout', float('nan')):.3f} | "
            f"{('%.4g' % e['tau']) if e.get('tau') is not None else '-'} |"
        )
    lines += ["", "## Primary verdict"]
    for p, v in verdict.items():
        lines.append(
            f"- {p}: passes A/B/C = {v['passes_ABC']} "
            f"(R2_train={v['r2_train']:.3f}, R2_heldout={v['r2_heldout']}, "
            f"norm-null R2_heldout={v['norm_null_r2_heldout']})"
        )
    lines += ["", "## Ranked by held-out R^2"]
    for zn, r2 in ranked:
        lines.append(f"- {zn}: {r2:.3f}")
    (args.out / "collapse.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}/collapse.json and collapse.md")


if __name__ == "__main__":
    main()
