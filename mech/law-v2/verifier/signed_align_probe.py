#!/usr/bin/env python3
"""law-v2 VERIFIER: signed-alignment probe.

The zoo's subsumption test (C1sat-H7kin+align) adds H7's alignment factor
1 + rho0*d^T*(M-1) with bounds rho0 in [0, 64], d in [0.05, 0.999] on top of
T/C.  Because M >= 1, that factor is >= 1 identically: it can only RAISE
momentum-arm predictions.  But the C1sat-H7kin held-out residuals show the
momentum arms are OVER-predicted at mid-T (mu0 rows high, momentum rows low,
Spearman-vs-mu rho = -0.467) -- the correction needed on top of T/C is
BELOW 1.  The subsumption test therefore could not have revived the
alignment term regardless of whether T/C is the right kinematics.

This probe adds a *signed* decaying momentum-arm correction on top of T/C:

    delta_bits = kappa * d^T * (M - 1)        (kappa free in sign, bits)

and, separately, a signed correction on the (C - T) excess-displacement
lever (closer to the phi structure the theory lane describes):

    delta_bits = kappa * d^T * log2(C/T)

If either revives with kappa < 0 and materially improves LOCO / removes the
mu structure, the "T/C subsumes H7's fitted alignment structure" claim is
refuted as evidence (the endpoint league comparison stands separately).

Run: .venv/bin/python mech/law-v2/verifier/signed_align_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "zoo"))
import fit_zoo as fz  # noqa: E402


def make_models() -> list[fz.Model]:
    zoo: list[fz.Model] = []

    def g_signedM(df, th):
        kappa, d, alpha, f, sigma, beta_c = th
        extra = np.where(df["conv"] == "nesterov_corrected", beta_c, 0.0)
        return (df["l2kin"] + kappa * d ** df["T"] * (df["M"] - 1.0)
                + np.log2(df["T"] ** (-alpha) + f)
                - extra * df["l2T"] + sigma * np.log2(df["S"] / 2560.0))
    zoo.append(fz.Model(
        "Vsigned-align-M",
        "T/C + signed kappa*d^T*(M-1) bits on momentum arms",
        ["kappa", "d", "alpha", "f", "sigma", "beta_c"],
        np.array([-4.0, 0.05, -0.5, 0.0, -1.0, -1.0]),
        np.array([4.0, 0.999, 2.5, 0.9, 1.0, 2.0]), g_signedM,
        n_global=5, n_stratum=4,
        stratum_note="3 scale intercepts + 1 corrected-arm tilt"))

    def g_signedC(df, th):
        kappa, d, alpha, f, sigma, beta_c = th
        lever = -df["l2kin"]  # log2(C/T) >= 0, grows with mu and T
        extra = np.where(df["conv"] == "nesterov_corrected", beta_c, 0.0)
        return (df["l2kin"] + kappa * d ** df["T"] * lever
                + np.log2(df["T"] ** (-alpha) + f)
                - extra * df["l2T"] + sigma * np.log2(df["S"] / 2560.0))
    zoo.append(fz.Model(
        "Vsigned-align-C",
        "T/C + signed kappa*d^T*log2(C/T) bits",
        ["kappa", "d", "alpha", "f", "sigma", "beta_c"],
        np.array([-4.0, 0.05, -0.5, 0.0, -1.0, -1.0]),
        np.array([4.0, 0.999, 2.5, 0.9, 1.0, 2.0]), g_signedC,
        n_global=5, n_stratum=4,
        stratum_note="3 scale intercepts + 1 corrected-arm tilt"))

    # a mu-shaped hump correction: kappa * (T/tau) * exp(1 - T/tau) * l2kin
    # (peaks at T = tau, vanishes at both ends -- the shape of the measured
    #  phi dip; still momentum-specific through l2kin).
    def g_hump(df, th):
        kappa, tau, alpha, f, sigma, beta_c = th
        x = df["T"] / tau
        hump = x * np.exp(1.0 - x)
        extra = np.where(df["conv"] == "nesterov_corrected", beta_c, 0.0)
        return (df["l2kin"] * (1.0 + kappa * hump)
                + np.log2(df["T"] ** (-alpha) + f)
                - extra * df["l2T"] + sigma * np.log2(df["S"] / 2560.0))
    zoo.append(fz.Model(
        "Vhump-phi",
        "T/C scaled by (1+kappa*(T/tau)e^{1-T/tau}) hump",
        ["kappa", "tau", "alpha", "f", "sigma", "beta_c"],
        np.array([-3.0, 1.0, -0.5, 0.0, -1.0, -1.0]),
        np.array([3.0, 120.0, 2.5, 0.9, 1.0, 2.0]), g_hump,
        n_global=5, n_stratum=4,
        stratum_note="3 scale intercepts + 1 corrected-arm tilt"))

    return zoo


def main() -> None:
    df = fz.load()
    results = {}
    for m in make_models():
        st = fz.full_fit_stats(m, df, fz.SEED)
        pooled, per_c, resid = fz.loco(m, df, fz.SEED)
        sp_mu = stats.spearmanr(df["mu"], resid)
        groups = [resid[(df["convention"] == c).to_numpy()]
                  for c in fz.CONVS if (df["convention"] == c).any()]
        kw = stats.kruskal(*groups)
        results[m.name] = dict(
            theta=dict(zip(m.theta_names, st["theta"])),
            rmse_full=st["rmse_full"], rmse_loco=pooled, k=m.k,
            aic=st["aic"], bic=st["bic"],
            spearman_mu=dict(rho=float(sp_mu.statistic),
                             p=float(sp_mu.pvalue)),
            kruskal_conv_p=float(kw.pvalue),
            per_campaign=per_c)
        print(f"{m.name:18s} LOCO={pooled:.3f} full={st['rmse_full']:.3f} "
              f"mu-rho={sp_mu.statistic:+.3f} (p={sp_mu.pvalue:.2g}) "
              f"conv-KW p={kw.pvalue:.2g}\n  theta={results[m.name]['theta']}")
    with open(HERE / "signed_align_results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("written:", HERE / "signed_align_results.json")


if __name__ == "__main__":
    main()
