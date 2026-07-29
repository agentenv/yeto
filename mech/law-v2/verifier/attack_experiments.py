#!/usr/bin/env python3
"""law-v2 VERIFIER: adversarial experiments against the C1 claim.

Reuses the zoo harness (mech/law-v2/zoo/fit_zoo.py) unchanged: same data,
same equal-point objective, same LOCO protocol, same fallbacks, same seed.

Attacks implemented here:
  A. SUBSUMPTION POWER (circularity): rerun the zoo's "alignment kept on top"
     subsumption test with deliberately WRONG kinematic factors
     (T/C)^gamma, gamma in {0.7, 1.3}, and with the kinematics deleted
     entirely (gamma=0).  If the pipeline also drives the alignment term to
     irrelevance and posts comparable LOCO under wrong kinematics, the
     "T/C subsumes H7's alignment structure" result has no discriminating
     power.  If the alignment term revives under wrong gamma, the
     subsumption test has teeth.
  B. SHAPE COMPETITOR (forensics consistency): score the forensics lane's
     saturating-exponential-in-log kinematics
       kin_F = (1-mu) * exp(ln(A_conv) * exp(-T/tau)),  raw/hb arms only,
     in the same harness -- both the near-parameter-free version
     (A = 1/(1-mu)) and the free-amplitude version -- against C1sat-H7kin.
  C. FLOOR PROVENANCE (leakage): leave BOTH T=160 campaigns (TP-v1, TP-v2)
     out simultaneously; the saturating-clock floor f was designed from
     their T=160 upturn, and ordinary LOCO always keeps one sibling in
     training.  Report held-out RMSE on the pair and the fitted f without
     them.
  D. SCALE AUDIT: per-point LOCO residuals at 1.7B/7B for H7 vs the
     C-variants (no refit; reads loco_residuals.json).

Run:  .venv/bin/python mech/law-v2/verifier/attack_experiments.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "zoo"))
import fit_zoo as fz  # noqa: E402

LOG2E = 1.0 / math.log(2.0)
OUT = HERE


def align_bits(rho0: float, d: float, M: np.ndarray, T: np.ndarray):
    return np.log2(1.0 + rho0 * d ** T * (M - 1.0))


def make_attack_models() -> list[fz.Model]:
    zoo: list[fz.Model] = []

    # ---- A. gamma-distorted kinematics + H7 alignment term on top ----
    def mk_gamma(gamma: float) -> fz.Model:
        def g(df, th):
            rho0, d, alpha, f, sigma, beta_c = th
            align = 1.0 + rho0 * d ** df["T"] * (df["M"] - 1.0)
            extra = np.where(df["conv"] == "nesterov_corrected", beta_c, 0.0)
            return (gamma * df["l2kin"] + np.log2(align)
                    + np.log2(df["T"] ** (-alpha) + f)
                    - extra * df["l2T"]
                    + sigma * np.log2(df["S"] / 2560.0))
        return fz.Model(
            f"Vgamma{gamma:.1f}+align",
            f"(T/C)^{gamma} wrong kinematics + H7 alignment kept on top",
            ["rho0", "d", "alpha", "f", "sigma", "beta_c"],
            np.array([0.0, 0.05, -0.5, 0.0, -1.0, -1.0]),
            np.array([64.0, 0.999, 2.5, 0.9, 1.0, 2.0]), g,
            n_global=5, n_stratum=4,
            stratum_note="3 scale intercepts + 1 corrected-arm tilt")

    for gamma in (0.0, 0.7, 1.3):
        zoo.append(mk_gamma(gamma))
    # gamma = 1.0 is exactly the zoo's C1sat-H7kin+align; refit for symmetry.
    zoo.append(mk_gamma(1.0))

    # gamma-distorted kinematics WITHOUT alignment (does LOCO even notice
    # a +-30% distortion of the kinematic exponent once the clock refits?)
    def mk_gamma_pure(gamma: float) -> fz.Model:
        def g(df, th):
            alpha, f, sigma, beta_c = th
            extra = np.where(df["conv"] == "nesterov_corrected", beta_c, 0.0)
            return (gamma * df["l2kin"]
                    + np.log2(df["T"] ** (-alpha) + f)
                    - extra * df["l2T"]
                    + sigma * np.log2(df["S"] / 2560.0))
        return fz.Model(
            f"Vgamma{gamma:.1f}-noalign",
            f"(T/C)^{gamma} wrong kinematics, no alignment term",
            ["alpha", "f", "sigma", "beta_c"],
            np.array([-0.5, 0.0, -1.0, -1.0]),
            np.array([2.5, 0.9, 1.0, 2.0]), g,
            n_global=3, n_stratum=4,
            stratum_note="3 scale intercepts + 1 corrected-arm tilt")

    for gamma in (0.0, 0.7, 1.3):
        zoo.append(mk_gamma_pure(gamma))

    # free-gamma: let the data pick the kinematic exponent (is gamma=1
    # actually preferred, or does the fit want a weaker transform?)
    def g_gfree(df, th):
        gamma, alpha, f, sigma, beta_c = th
        extra = np.where(df["conv"] == "nesterov_corrected", beta_c, 0.0)
        return (gamma * df["l2kin"]
                + np.log2(df["T"] ** (-alpha) + f)
                - extra * df["l2T"]
                + sigma * np.log2(df["S"] / 2560.0))
    zoo.append(fz.Model(
        "Vgamma-free", "(T/C)^gamma with gamma a free global parameter",
        ["gamma", "alpha", "f", "sigma", "beta_c"],
        np.array([-0.5, -0.5, 0.0, -1.0, -1.0]),
        np.array([2.5, 2.5, 0.9, 1.0, 2.0]), g_gfree,
        n_global=4, n_stratum=4,
        stratum_note="3 scale intercepts + 1 corrected-arm tilt"))

    # ---- B. forensics saturating-exponential kinematics ----
    def kinF(df, tau: float, lnA_raw: float, lnA_hb: float) -> np.ndarray:
        decay = np.exp(-df["T"] / tau)
        add = np.zeros(len(df["T"]))
        mraw = df["conv"] == "nesterov_raw"
        mhb = df["conv"] == "heavy_ball"
        add[mraw] = lnA_raw * decay[mraw] * LOG2E
        add[mhb] = lnA_hb * decay[mhb] * LOG2E
        return df["l1m"] + add  # mu0: l1m=0; corrected: (1-mu) as in C1/H7

    # near-parameter-free: A = 1/(1-mu) (steady multiplier), tau shared free
    def g_FsatA0(df, th):
        tau, alpha, f, sigma, beta_c = th
        lnA = -np.log(np.maximum(1.0 - df["mu"], 1e-9))
        decay = np.exp(-df["T"] / tau)
        mom = ((df["conv"] == "nesterov_raw")
               | (df["conv"] == "heavy_ball"))
        add = np.where(mom, lnA * decay * LOG2E, 0.0)
        extra = np.where(df["conv"] == "nesterov_corrected", beta_c, 0.0)
        return (df["l1m"] + add + np.log2(df["T"] ** (-alpha) + f)
                - extra * df["l2T"] + sigma * np.log2(df["S"] / 2560.0))
    zoo.append(fz.Model(
        "VFsat-fixedA",
        "forensics kinematics, A=1/(1-mu) fixed, shared tau",
        ["tau", "alpha", "f", "sigma", "beta_c"],
        np.array([0.5, -0.5, 0.0, -1.0, -1.0]),
        np.array([80.0, 2.5, 0.9, 1.0, 2.0]), g_FsatA0,
        n_global=4, n_stratum=4,
        stratum_note="3 scale intercepts + 1 corrected-arm tilt"))

    # free amplitudes per momentum convention (fallback: hb <- raw)
    def g_Fsat(df, th):
        tau, lnAr, lnAh, alpha, f, sigma, beta_c = th
        extra = np.where(df["conv"] == "nesterov_corrected", beta_c, 0.0)
        return (kinF(df, tau, lnAr, lnAh)
                + np.log2(df["T"] ** (-alpha) + f)
                - extra * df["l2T"] + sigma * np.log2(df["S"] / 2560.0))
    def fb_Fsat(train, th):
        if not (train["conv"] == "heavy_ball").any():
            th[2] = th[1]
        return th
    zoo.append(fz.Model(
        "VFsat-freeA",
        "forensics kinematics, free lnA_raw/lnA_hb, shared tau",
        ["tau", "lnA_raw", "lnA_hb", "alpha", "f", "sigma", "beta_c"],
        np.array([0.5, 0.0, 0.0, -0.5, 0.0, -1.0, -1.0]),
        np.array([80.0, 4.0, 4.0, 2.5, 0.9, 1.0, 2.0]), g_Fsat,
        n_global=6, n_stratum=4, fallback=fb_Fsat,
        stratum_note="3 scale intercepts + 1 corrected-arm tilt"))

    return zoo


def reference_models() -> dict[str, fz.Model]:
    zoo: list[fz.Model] = fz.make_zoo()
    fz.add_own_designs(zoo)
    fz.add_theory_candidates(zoo)
    return {m.name: m for m in zoo}


def main() -> None:
    df = fz.load()
    refs = reference_models()
    results: dict[str, dict] = {}

    # ---------------- A + B: score attack models ----------------
    attack = make_attack_models()
    for m in attack + [refs["C1sat-H7kin"], refs["H7-rho-floor-S"],
                       refs["C1sat-H7kin+align"]]:
        st = fz.full_fit_stats(m, df, fz.SEED)
        pooled, per_c, resid = fz.loco(m, df, fz.SEED)
        entry = dict(theta=dict(zip(m.theta_names, st["theta"])),
                     rmse_full=st["rmse_full"], rmse_loco=pooled,
                     aic=st["aic"], bic=st["bic"], k=m.k,
                     per_campaign=per_c)
        # alignment-term footprint (bits) if present
        if "rho0" in m.theta_names and "d" in m.theta_names:
            th = dict(zip(m.theta_names, st["theta"]))
            dp = fz.pack(df)
            ab = align_bits(th["rho0"], th["d"], dp["M"], dp["T"])
            entry["align_bits_max"] = float(np.abs(ab).max())
            entry["align_bits_at_T2_mu09_raw"] = float(np.log2(
                1.0 + th["rho0"] * th["d"] ** 2
                * (fz.finite_age_M(2, 0.9, "nesterov_raw") - 1.0)))
        results[m.name] = entry
        print(f"{m.name:24s} LOCO={pooled:.3f} full={st['rmse_full']:.3f} "
              f"k={m.k} theta={entry['theta']}")
        if "align_bits_max" in entry:
            print(f"{'':24s}   align footprint: max |bits| = "
                  f"{entry['align_bits_max']:.3f}")

    # ---------------- C: leave-both-TP-out floor probe ----------------
    print("\n-- C: leave-{TP-v1,TP-v2}-out (both T=160 campaigns) --")
    pair = ["TP-v1", "TP-v2"]
    tr = df[~df["campaign"].isin(pair)]
    te = df[df["campaign"].isin(pair)]
    trp, tep = fz.pack(tr), fz.pack(te)
    floor_probe = {}
    for name in ["H7-rho-floor-S", "C1sat-H7kin", "C1sat-pure", "C1sat+C3"]:
        m = refs[name]
        rng = np.random.default_rng(fz.SEED)
        th = fz.fit(m, trp, rng)
        pred = fz.predict(m, th, trp, tep)
        r = te["y"].to_numpy() - pred
        t160 = te["T"].to_numpy() == 160
        floor_probe[name] = dict(
            theta=dict(zip(m.theta_names, map(float, th))),
            rmse_pair=float(np.sqrt(np.mean(r ** 2))),
            rmse_T160=float(np.sqrt(np.mean(r[t160] ** 2))),
            mean_T160=float(np.mean(r[t160])),
            n_T160=int(t160.sum()))
        print(f"{name:16s} pair RMSE={floor_probe[name]['rmse_pair']:.3f} "
              f"T160 RMSE={floor_probe[name]['rmse_T160']:.3f} "
              f"T160 mean={floor_probe[name]['mean_T160']:+.3f} "
              f"theta={floor_probe[name]['theta']}")
    results["floor_probe_leave_TPv1_TPv2_out"] = floor_probe

    # ---------------- D: 1.7B / 7B point-level LOCO residuals ----------
    print("\n-- D: non-135M LOCO residuals (from zoo loco_residuals.json) --")
    with open(HERE.parent / "zoo" / "loco_residuals.json") as f:
        lr = json.load(f)
    mask = df["scale"] != "135M"
    tab = []
    for _, row in df[mask].iterrows():
        i = row.name
        tab.append(dict(
            campaign=row["campaign"], scale=row["scale"],
            conv=row["convention"], T=int(row["T"]), mu=float(row["mu"]),
            H7=lr["H7-rho-floor-S"][i], C1satH7=lr["C1sat-H7kin"][i],
            C1satC3=lr["C1sat+C3"][i]))
        print(f"{row['campaign']:5s} {row['scale']:4s} "
              f"{row['convention']:18s} T={int(row['T']):3d} mu={row['mu']:.2f}"
              f"  H7={lr['H7-rho-floor-S'][i]:+.3f}"
              f"  C1sat-H7kin={lr['C1sat-H7kin'][i]:+.3f}"
              f"  C1sat+C3={lr['C1sat+C3'][i]:+.3f}")
    results["non135M_loco_residuals"] = tab

    with open(OUT / "attack_results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("\nwritten:", OUT / "attack_results.json")


if __name__ == "__main__":
    main()
