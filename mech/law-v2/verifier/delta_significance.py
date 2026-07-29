#!/usr/bin/env python3
"""Jackknife SEs + residual-structure tests for the verifier's competitor
models vs C1sat-H7kin and H7."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "zoo"))
import fit_zoo as fz  # noqa: E402
import attack_experiments as ax  # noqa: E402


def main() -> None:
    df = fz.load()
    refs = ax.reference_models()
    attack = {m.name: m for m in ax.make_attack_models()}
    names = ["C1sat-H7kin", "H7-rho-floor-S"]
    models = [refs[n] for n in names]
    models += [attack["VFsat-fixedA"], attack["Vgamma-free"]]

    resids = {}
    for m in models:
        pooled, per_c, resid = fz.loco(m, df, fz.SEED)
        resids[m.name] = resid
        sp_T = stats.spearmanr(df["T"], resid)
        sp_mu = stats.spearmanr(df["mu"], resid)
        sp_S = stats.spearmanr(df["S"], resid)
        groups = [resid[(df["convention"] == c).to_numpy()]
                  for c in fz.CONVS if (df["convention"] == c).any()]
        kw = stats.kruskal(*groups)
        print(f"{m.name:16s} LOCO={pooled:.3f}  "
              f"T rho={sp_T.statistic:+.2f}(p={sp_T.pvalue:.2g}) "
              f"mu rho={sp_mu.statistic:+.2f}(p={sp_mu.pvalue:.2g}) "
              f"S rho={sp_S.statistic:+.2f}(p={sp_S.pvalue:.2g}) "
              f"convKW p={kw.pvalue:.2g}")

    print()
    out = {}
    for a in ["VFsat-fixedA", "Vgamma-free"]:
        for b in ["C1sat-H7kin", "H7-rho-floor-S"]:
            d, se = fz.jackknife_delta(resids[a], resids[b], df)
            out[f"{a} - {b}"] = dict(delta=d, jk_se=se)
            print(f"{a} - {b}: delta={d:+.3f} bits, jk SE={se:.3f} "
                  f"({abs(d)/se:.1f}x)")
    d, se = fz.jackknife_delta(resids["C1sat-H7kin"],
                               resids["H7-rho-floor-S"], df)
    out["C1sat-H7kin - H7"] = dict(delta=d, jk_se=se)
    print(f"C1sat-H7kin - H7: delta={d:+.3f}, jk SE={se:.3f}")
    with open(HERE / "delta_significance.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
