#!/usr/bin/env python
"""law-v2 forensics: empirical age transform r(T) per stratum.

Pure characterization of the matched-pair ratio
    r(T; mu, convention) = [eta*(mu,T) / eta*(0,T)] / (1-mu)
across all campaigns with matched pairs, plus shape-hypothesis tests,
S/scale residual checks, plots, and tidy CSVs.

No theory. Inputs: mech/law-unification/tuned_optima.csv (+ cross-check
against mech/law-unification/paired_cancellation.csv).
Outputs land in mech/law-v2/forensics/.
"""
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LU = os.path.normpath(os.path.join(HERE, "..", "..", "law-unification"))
Z95 = 1.959963984540054
LOG2 = math.log(2.0)

# ----------------------------------------------------------------------
# chi-square survival function without scipy (regularized upper incomplete
# gamma, Numerical Recipes gser/gcf).
# ----------------------------------------------------------------------

def _gser(a, x, itmax=500, eps=3e-12):
    gln = math.lgamma(a)
    ap = a
    s = 1.0 / a
    delt = s
    for _ in range(itmax):
        ap += 1.0
        delt *= x / ap
        s += delt
        if abs(delt) < abs(s) * eps:
            break
    return s * math.exp(-x + a * math.log(x) - gln)


def _gcf(a, x, itmax=500, eps=3e-12):
    gln = math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x, k):
    """P(Chi2_k > x)."""
    if k <= 0:
        return float("nan")
    if x <= 0:
        return 1.0
    a, xx = 0.5 * k, 0.5 * x
    if xx > 700 + a:  # deep tail: log-scale continued fraction would underflow anyway
        return 0.0
    if xx < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gser(a, xx)))
    return max(0.0, min(1.0, _gcf(a, xx)))


# ----------------------------------------------------------------------
# Load ledger, build matched pairs
# ----------------------------------------------------------------------

opt = pd.read_csv(os.path.join(LU, "tuned_optima.csv"))
pc = pd.read_csv(os.path.join(LU, "paired_cancellation.csv"))

# Sanity: H == S/T identically? (forensic fact used throughout the report)
h_eq = np.allclose(opt["H"] * opt["T"], opt["S"])
print(f"[check] H*T == S for all {len(opt)} points: {h_eq}")


def M_code(convention, mu, T):
    if convention == "nesterov_raw":
        return (1.0 - mu ** (T + 1)) / (1.0 - mu)
    if convention == "heavy_ball":
        return (1.0 - mu ** T) / (1.0 - mu)
    return 1.0  # nesterov_corrected, mu0


mom = opt[opt["mu"] > 0].copy()
ctl = opt[opt["mu"] == 0].copy()

# G5B snoo-* arms are a separate optimizer family with no mu0 control at the
# same convention; they never pair (matches the existing 62-row ledger).
mom = mom[~mom["arm"].str.startswith("snoo")]

key_in = ["campaign", "scale", "T", "H", "S"]
key_x = ["scale", "T", "H", "S"]

rows = []
for _, m in mom.iterrows():
    cands = ctl[(ctl["campaign"] == m["campaign"]) & (ctl["scale"] == m["scale"]) &
                (ctl["T"] == m["T"]) & (ctl["H"] == m["H"]) & (ctl["S"] == m["S"])]
    pair_type = "within_campaign"
    if len(cands) == 0:
        # secondary tier: same scale/T/H/S from another campaign (135M TP family
        # + G-series share identical recipes). Campaign effects do NOT cancel.
        cands = ctl[(ctl["scale"] == m["scale"]) & (ctl["T"] == m["T"]) &
                    (ctl["H"] == m["H"]) & (ctl["S"] == m["S"])]
        pair_type = "cross_campaign"
    if len(cands) == 0:
        print(f"[warn] no control for {m['point_id']}")
        continue
    mu = m["mu"]
    conv = m["convention"]
    if pair_type == "within_campaign" and len(cands) == 1:
        pair_ctls = [cands.iloc[0]]
    else:
        # duplicate mu0 lanes in-cell (TP-v3 ran mu0 in both raw and corrected
        # lanes) or cross-campaign controls: pool by IVW -> one pair per
        # momentum point.
        pair_ctls = ["POOLED"]
    for c in pair_ctls:
        if isinstance(c, str):  # pooled cross-campaign control: IVW mean log eta
            w = 1.0 / np.maximum(cands["se_log_eta"].to_numpy(), 1e-6) ** 2
            log_eta_c = float((np.log(cands["eta_star"]) * w).sum() / w.sum())
            # conservative: pooled SE cannot be tighter than the best single control
            se_c = max(math.sqrt(1.0 / w.sum()), float(cands["se_log_eta"].min()))
            eta_c = math.exp(log_eta_c)
            ci_lo_c = float((cands["eta_ci_low"] / cands["eta_star"]).min()) * eta_c
            ci_hi_c = float((cands["eta_ci_high"] / cands["eta_star"]).max()) * eta_c
            c_campaign = ";".join(sorted(cands["campaign"]))
            c_id = ";".join(sorted(cands["point_id"]))
            c_ok = bool(cands["noise_eligible"].all())
        else:
            eta_c, se_c = c["eta_star"], c["se_log_eta"]
            ci_lo_c, ci_hi_c = c["eta_ci_low"], c["eta_ci_high"]
            c_campaign, c_id, c_ok = c["campaign"], c["point_id"], bool(c["noise_eligible"])
        ratio = m["eta_star"] / eta_c
        r = ratio / (1.0 - mu)
        se = math.sqrt(m["se_log_eta"] ** 2 + se_c ** 2)
        # conservative marginal-endpoint CI (same convention as paired ledger)
        r_lo = (m["eta_ci_low"] / ci_hi_c) / (1.0 - mu)
        r_hi = (m["eta_ci_high"] / ci_lo_c) / (1.0 - mu)
        Mv = M_code(conv, mu, int(m["T"]))
        rows.append(dict(
            stratum=f"{conv}:mu{mu:g}", convention=conv, mu=mu,
            campaign=m["campaign"], control_campaign=c_campaign,
            pair_type=pair_type, scale=m["scale"], T=int(m["T"]),
            H=int(m["H"]), S=int(m["S"]),
            eta_ratio=ratio, r=r, r_ci_lo=r_lo, r_ci_hi=r_hi,
            se_log_r=se, M_code=Mv, r_over_M=r / Mv,
            log2_r=math.log2(r), log2_r_over_M=math.log2(r / Mv),
            noise_eligible=bool(m["noise_eligible"]) and c_ok,
            momentum_point_id=m["point_id"], control_point_id=c_id,
        ))

pairs = pd.DataFrame(rows).sort_values(
    ["convention", "mu", "T", "scale", "S", "campaign"]).reset_index(drop=True)

n_within = (pairs["pair_type"] == "within_campaign").sum()
print(f"[pairs] {len(pairs)} total ({n_within} within-campaign, "
      f"{len(pairs) - n_within} cross-campaign)")

# duplicate-control forensics: TP-v3 ran mu=0 twice (raw + corrected lanes,
# same optimizer). Their tuned-rate disagreement is a direct empirical measure
# of pure re-tuning noise at mu=0.
dup = ctl.groupby(key_in).filter(lambda g: len(g) > 1)
if len(dup):
    print("[dup-controls] duplicate mu0 lanes (pure re-tuning noise):")
    for k, g in dup.groupby(key_in):
        e = g["eta_star"].to_numpy()
        d_bits = math.log2(e.max() / e.min())
        print(f"  {k}: etas={list(np.round(e, 6))} spread={d_bits:.3f} bits "
              f"(bootstrap se_log sum={g['se_log_eta'].sum():.4f})")

# cross-check within-campaign pairs against the published 62-row ledger
pcheck = pairs[pairs["pair_type"] == "within_campaign"].merge(
    pc[["momentum_point_id", "observed_to_law_ratio"]],
    on="momentum_point_id", how="outer", indicator=True)
n_both = (pcheck["_merge"] == "both").sum()
print(f"[check] within-campaign momentum points matching published ledger: "
      f"{n_both}/{len(pc)} (mine-only={(pcheck['_merge'] == 'left_only').sum()}, "
      f"pub-only={(pcheck['_merge'] == 'right_only').sum()})")
both = pcheck[pcheck["_merge"] == "both"]
mism = (both["r_over_M"] / both["observed_to_law_ratio"] - 1).abs()
print(f"[check] r/M vs published observed_to_law rel. diff: "
      f"non-TP-v3 max={mism[~both['momentum_point_id'].str.startswith('TP-v3')].max():.2e}, "
      f"TP-v3 (pooled dual mu0 control vs corrected-only) max={mism[both['momentum_point_id'].str.startswith('TP-v3')].max():.2e}")

pairs.to_csv(os.path.join(HERE, "r_of_T_pairs.csv"), index=False)

# ----------------------------------------------------------------------
# Replicate noise floor per stratum (within-cell scatter of log r across
# S-replicates / campaigns at the same (stratum, scale, T)) — bootstrap SEs
# are known to be far too tight (heterogeneity chi2=2.8e6 upstream).
# ----------------------------------------------------------------------

def replicate_floor(df):
    """Pooled within-(T,scale) std of log r beyond bootstrap SEs."""
    num = 0.0
    dof = 0
    se2 = []
    for _, g in df.groupby(["T", "scale"]):
        if len(g) < 2:
            continue
        x = g["log2_r"].to_numpy() * LOG2
        num += ((x - x.mean()) ** 2).sum()
        dof += len(g) - 1
        se2 += list(g["se_log_r"] ** 2)
    if dof == 0:
        return float("nan")
    tot = num / dof
    boot = float(np.mean(se2)) if se2 else 0.0
    return math.sqrt(max(tot - boot, 0.0))


strata = {}
for (conv, mu), g in pairs.groupby(["convention", "mu"]):
    fit_g = g[(g["pair_type"] == "within_campaign") & (g["scale"] == "135M")]
    if len(fit_g) == 0:  # G9A has no mu0 control at all -> only cross pairs
        fit_g = g[g["scale"] == "135M"]
    s_rep = replicate_floor(fit_g)
    strata[(conv, mu)] = dict(fit=fit_g, all=g, s_rep=s_rep)
    print(f"[stratum] {conv} mu={mu:g}: n_fit={len(fit_g)} ages="
          f"{sorted(fit_g['T'].unique())} s_rep={s_rep:.4f} (nat log)")

# global fallback floor: median of estimable strata
floors = [v["s_rep"] for v in strata.values() if np.isfinite(v["s_rep"])]
S_REP_FALLBACK = float(np.median(floors))
print(f"[floor] fallback replicate floor {S_REP_FALLBACK:.4f}")

# ----------------------------------------------------------------------
# Shape models. All fit in natural-log space: y = log r, x = T.
#   m(T) = log M_code(T)  (convention-specific, code-true multiplier)
# ----------------------------------------------------------------------

def wls(X, y, w):
    Xw = X * w[:, None]
    beta, *_ = np.linalg.lstsq(Xw, y * w, rcond=None)
    resid = y - X @ beta
    return beta, resid


def fit_model(name, T, y, sig, mvec):
    """Return dict with params, chi2, k. y=log r, sig=eff sigma, mvec=log M."""
    w = 1.0 / sig
    n = len(y)
    ones = np.ones(n)

    if name == "law_pure_multiplier":
        resid = y - mvec
        return dict(k=0, resid=resid, params={})
    if name == "scaled_multiplier":
        beta, resid = wls(np.column_stack([ones]), y - mvec, w)
        return dict(k=1, resid=resid, params={"c": math.exp(beta[0])})
    if name == "constant":
        beta, resid = wls(np.column_stack([ones]), y, w)
        return dict(k=1, resid=resid, params={"c": math.exp(beta[0])})
    if name == "exp_decay":
        beta, resid = wls(np.column_stack([ones, T.astype(float)]), y, w)
        return dict(k=2, resid=resid,
                    params={"c": math.exp(beta[0]), "rho": math.exp(beta[1])})
    if name == "multiplier_times_decay":
        beta, resid = wls(np.column_stack([ones, T.astype(float)]), y - mvec, w)
        return dict(k=2, resid=resid,
                    params={"c": math.exp(beta[0]), "rho": math.exp(beta[1])})
    if name == "power_law":
        beta, resid = wls(np.column_stack([ones, np.log(T.astype(float))]), y, w)
        return dict(k=2, resid=resid,
                    params={"c": math.exp(beta[0]), "a": -beta[1]})
    if name == "sat_exp_to_one":
        # log r = B*exp(-T/tau): r saturates to exactly 1; grid tau
        best = None
        for tau in np.geomspace(0.5, 400.0, 240):
            X = np.exp(-T / tau)[:, None]
            beta, resid = wls(X, y, w)
            c2 = float(((resid / sig) ** 2).sum())
            if best is None or c2 < best[0]:
                best = (c2, tau, beta, resid)
        _, tau, beta, resid = best
        return dict(k=2, resid=resid,
                    params={"r0_amp": math.exp(beta[0]), "tau": tau})
    if name == "saturating_exp":
        # log r = A + B*exp(-T/tau); grid tau
        best = None
        for tau in np.geomspace(0.5, 400.0, 240):
            X = np.column_stack([ones, np.exp(-T / tau)])
            beta, resid = wls(X, y, w)
            c2 = float(((resid / sig) ** 2).sum())
            if best is None or c2 < best[0]:
                best = (c2, tau, beta, resid)
        _, tau, beta, resid = best
        return dict(k=3, resid=resid,
                    params={"r_inf": math.exp(beta[0]),
                            "r0_amp": math.exp(beta[1]), "tau": tau})
    if name == "two_regime":
        # log r = c + m(min(T,Tc)) + log(rho)*max(0, T-Tc); grid Tc
        best = None
        for Tc in np.geomspace(1.0, 300.0, 300):
            mclip = np.array([_logM_cont(min(t, Tc)) for t in T])
            X = np.column_stack([ones, np.maximum(0.0, T - Tc)])
            beta, resid = wls(X, y - mclip, w)
            c2 = float(((resid / sig) ** 2).sum())
            if best is None or c2 < best[0]:
                best = (c2, Tc, beta, resid)
        _, Tc, beta, resid = best
        return dict(k=3, resid=resid,
                    params={"c": math.exp(beta[0]), "Tc": Tc,
                            "rho": math.exp(beta[1])})
    raise ValueError(name)


MODELS = ["law_pure_multiplier", "scaled_multiplier", "constant", "exp_decay",
          "multiplier_times_decay", "power_law", "sat_exp_to_one",
          "saturating_exp", "two_regime"]

fit_rows = []
curves = {}  # (conv,mu) -> dict for plotting

for (conv, mu), st in strata.items():
    g = st["fit"]
    ages = sorted(g["T"].unique())
    if len(ages) < 2:
        print(f"[skip] {conv} mu={mu:g}: only {len(ages)} age(s)")
        continue
    T = g["T"].to_numpy(float)
    y = g["log2_r"].to_numpy() * LOG2
    s_rep = st["s_rep"] if np.isfinite(st["s_rep"]) else S_REP_FALLBACK
    sig_raw = g["se_log_r"].to_numpy(float)
    sig_raw = np.maximum(sig_raw, 1e-6)
    sig_eff = np.sqrt(sig_raw ** 2 + s_rep ** 2)

    def _logM_cont(t, conv=conv, mu=mu):
        return math.log(M_code(conv, mu, t)) if conv in (
            "nesterov_raw", "heavy_ball") else 0.0
    globals()["_logM_cont"] = _logM_cont
    mvec = np.array([_logM_cont(t) for t in T])

    st_curves = {}
    for name in MODELS:
        n_ages = len(ages)
        res = fit_model(name, T, y, sig_eff, mvec)
        k = res["k"]
        if n_ages < k:  # cannot even identify shape from distinct ages
            continue
        resid = res["resid"]
        n = len(y)
        chi2_eff = float(((resid / sig_eff) ** 2).sum())
        chi2_raw = float(((resid / sig_raw) ** 2).sum())
        dof = n - k
        aicc = chi2_eff + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else float("inf"))
        fit_rows.append(dict(
            convention=conv, mu=mu, model=name, n=n, n_ages=n_ages, k=k,
            dof=dof, chi2_eff=chi2_eff, chi2_eff_per_dof=chi2_eff / dof if dof > 0 else float("nan"),
            p_eff=chi2_sf(chi2_eff, dof) if dof > 0 else float("nan"),
            chi2_raw=chi2_raw, p_raw=chi2_sf(chi2_raw, dof) if dof > 0 else float("nan"),
            aicc=aicc, rmse_bits=float(np.sqrt(np.mean(resid ** 2))) / LOG2,
            saturated=(dof <= 0), s_rep_nat=s_rep,
            params=json.dumps({kk: round(vv, 6) for kk, vv in res["params"].items()}),
        ))
        st_curves[name] = res["params"]
    curves[(conv, mu)] = dict(params=st_curves, s_rep=s_rep)

fits = pd.DataFrame(fit_rows)
fits.to_csv(os.path.join(HERE, "shape_fits.csv"), index=False)

# ----------------------------------------------------------------------
# Cell means (inverse-variance with floor) for the tidy per-stratum table
# ----------------------------------------------------------------------
cm_rows = []
for (conv, mu), st in strata.items():
    g = st["all"]
    s_rep = st["s_rep"] if np.isfinite(st["s_rep"]) else S_REP_FALLBACK
    for (T, scale), c in g.groupby(["T", "scale"]):
        y = c["log2_r"].to_numpy() * LOG2
        se_in = np.nan_to_num(c["se_log_r"].to_numpy(), nan=S_REP_FALLBACK)
        sig = np.sqrt(se_in ** 2 + s_rep ** 2)
        w = 1.0 / sig ** 2
        mean = float((y * w).sum() / w.sum())
        se = math.sqrt(1.0 / w.sum())
        cm_rows.append(dict(
            convention=conv, mu=mu, scale=scale, T=T, n_pairs=len(c),
            r_mean=math.exp(mean), r_lo=math.exp(mean - Z95 * se),
            r_hi=math.exp(mean + Z95 * se), log2_r_mean=mean / LOG2,
            se_log2_r=se / LOG2, M_code=M_code(conv, mu, int(T)),
            log2_r_over_M=mean / LOG2 - math.log2(M_code(conv, mu, int(T))),
            campaigns=";".join(sorted(c["campaign"].unique())),
            pair_types=";".join(sorted(c["pair_type"].unique())),
        ))
cells = pd.DataFrame(cm_rows).sort_values(["convention", "mu", "scale", "T"])
cells.to_csv(os.path.join(HERE, "r_of_T_cellmeans.csv"), index=False)

# ----------------------------------------------------------------------
# S-dependence (G6 factorial only: the single unconfounded design) and
# scale-dependence (1.7B / 7B pairs vs 135M curve)
# ----------------------------------------------------------------------
s_rows = []
for conv in ["nesterov_raw", "nesterov_corrected"]:
    g6 = pairs[(pairs["campaign"] == "G6") & (pairs["convention"] == conv)]
    if len(g6) == 0:
        continue
    Ts = sorted(g6["T"].unique())
    y = g6["log2_r"].to_numpy() * LOG2
    x = np.log2(g6["S"].to_numpy() / 2560.0)
    D = np.column_stack([ (g6["T"].to_numpy() == t).astype(float) for t in Ts] + [x])
    s_rep = strata[(conv, 0.9)]["s_rep"]
    sig = np.sqrt(g6["se_log_r"].to_numpy() ** 2 + (s_rep if np.isfinite(s_rep) else S_REP_FALLBACK) ** 2)
    beta, resid = wls(D, y, 1.0 / sig)
    # slope covariance
    Xw = D / sig[:, None]
    cov = np.linalg.inv(Xw.T @ Xw)
    slope = beta[-1] / LOG2  # bits per doubling of S (== per doubling of H at fixed T)
    slope_se = math.sqrt(cov[-1, -1]) / LOG2
    s_rows.append(dict(convention=conv, mu=0.9, design="G6 factorial (T FE)",
                       n=len(g6), bits_per_doubling_S=slope,
                       ci_lo=slope - Z95 * slope_se, ci_hi=slope + Z95 * slope_se,
                       z=slope / slope_se,
                       p=2 * 0.5 * math.erfc(abs(slope / slope_se) / math.sqrt(2))))
s_dep = pd.DataFrame(s_rows)
s_dep.to_csv(os.path.join(HERE, "s_dependence.csv"), index=False)
print(s_dep.to_string())

# model selection helper: AICc is undefined/inf for k>=2 at n=3, so rank by
# goodness-of-fit p (dof>=1 models only), tie-broken by chi2/dof.
def pick_best(fsub):
    ok = fsub[fsub["dof"] >= 1]
    if len(ok) == 0:
        return None
    return ok.sort_values(["p_eff", "chi2_eff_per_dof"],
                          ascending=[False, True]).iloc[0]


# scale residuals: non-135M pairs vs best 135M model of their stratum


def model_curve(conv, mu, name, params, Tg):
    def mfun(t):
        return math.log(M_code(conv, mu, t)) if conv in ("nesterov_raw", "heavy_ball") else 0.0
    out = []
    for t in Tg:
        if name == "law_pure_multiplier":
            v = mfun(t)
        elif name == "scaled_multiplier":
            v = math.log(params["c"]) + mfun(t)
        elif name == "constant":
            v = math.log(params["c"])
        elif name == "exp_decay":
            v = math.log(params["c"]) + t * math.log(params["rho"])
        elif name == "multiplier_times_decay":
            v = math.log(params["c"]) + mfun(t) + t * math.log(params["rho"])
        elif name == "power_law":
            v = math.log(params["c"]) - params["a"] * math.log(t)
        elif name == "sat_exp_to_one":
            v = math.log(params["r0_amp"]) * math.exp(-t / params["tau"])
        elif name == "saturating_exp":
            v = math.log(params["r_inf"]) + math.log(params["r0_amp"]) * math.exp(-t / params["tau"])
        elif name == "two_regime":
            v = math.log(params["c"]) + mfun(min(t, params["Tc"])) + \
                math.log(params["rho"]) * max(0.0, t - params["Tc"])
        out.append(v / LOG2)  # log2
    return np.array(out)


sc_rows = []
for (conv, mu), st in strata.items():
    non135 = st["all"][st["all"]["scale"] != "135M"]
    if len(non135) == 0 or (conv, mu) not in curves:
        continue
    best = pick_best(fits[(fits["convention"] == conv) & (fits["mu"] == mu)])
    if best is None:
        continue
    params = json.loads(best["params"])
    for _, p in non135.iterrows():
        pred = model_curve(conv, mu, best["model"], params, [p["T"]])[0]
        sc_rows.append(dict(convention=conv, mu=mu, scale=p["scale"], T=p["T"],
                            campaign=p["campaign"], best_model=best["model"],
                            log2_r=p["log2_r"], log2_r_pred_135M=pred,
                            residual_bits=p["log2_r"] - pred,
                            momentum_point_id=p["momentum_point_id"]))
scale_res = pd.DataFrame(sc_rows)
scale_res.to_csv(os.path.join(HERE, "scale_residuals.csv"), index=False)
print(scale_res.to_string())

# ----------------------------------------------------------------------
# Plots (light-mode PNGs; fixed-order categorical palette)
# ----------------------------------------------------------------------
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
       "#4a3aa7", "#e34948"]
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.grid": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.edgecolor": "#c9c9c9", "axes.linewidth": 0.8,
    "axes.titlesize": 10, "figure.facecolor": "white",
})

# --- Fig 1: raw mu0.9, the richest stratum -----------------------------
conv, mu = "nesterov_raw", 0.9
g = strata[(conv, mu)]["all"]
fig, ax = plt.subplots(figsize=(7.2, 5.0))
camps = sorted(g["campaign"].unique())
cmap = {c: PAL[i % len(PAL)] for i, c in enumerate(camps)}
for c in camps:
    sub = g[g["campaign"] == c]
    for _, p in sub.iterrows():
        mk = {"135M": "o", "1.7B": "s", "7B": "D"}[p["scale"]]
        ax.errorbar(p["T"], p["r"],
                    yerr=[[p["r"] - p["r_ci_lo"]], [p["r_ci_hi"] - p["r"]]],
                    fmt=mk, ms=5 if p["scale"] == "135M" else 7,
                    mfc=cmap[c], mec="white", mew=0.8, ecolor=cmap[c],
                    elinewidth=1.0, capsize=0, zorder=3,
                    alpha=0.55 if p["pair_type"] == "cross_campaign" else 1.0)
Tg = np.geomspace(1.5, 220, 300)
Mg = np.array([M_code(conv, mu, t) for t in Tg])
ax.plot(Tg, Mg, color="#666666", lw=1.6, ls="--", zorder=2)
ax.annotate("law: r = M(T)  (code-true multiplier)", xy=(3.2, 4.4),
            color="#555555", fontsize=8)
best_models = ["power_law", "multiplier_times_decay", "two_regime", "sat_exp_to_one"]
mk_style = {"power_law": ("-", 2.0), "multiplier_times_decay": (":", 1.8),
            "two_regime": ("-.", 1.6), "sat_exp_to_one": ((0, (4, 2)), 1.4)}
fsub = fits[(fits["convention"] == conv) & (fits["mu"] == mu)]
for i, name in enumerate(best_models):
    row = fsub[fsub["model"] == name]
    if len(row) == 0:
        continue
    params = json.loads(row.iloc[0]["params"])
    yv = 2.0 ** model_curve(conv, mu, name, params, Tg)
    ls, lw = mk_style[name]
    ax.plot(Tg, yv, ls=ls, lw=lw, color="#333333", alpha=0.85 - 0.15 * i,
            label=f"{name} (chi2/dof={row.iloc[0]['chi2_eff_per_dof']:.1f})")
ax.axhline(1.0, color="#bbbbbb", lw=0.8, zorder=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("outer age T (rounds)")
ax.set_ylabel(r"r(T) = [$\eta^*(\mu,T)/\eta^*(0,T)$] / (1-$\mu$)")
ax.set_title("nesterov_raw, mu=0.9: matched-pair age transform vs shape hypotheses")
h1 = [plt.Line2D([], [], marker="o", ls="", mfc=cmap[c], mec="white", label=c) for c in camps]
h2 = [plt.Line2D([], [], marker=m, ls="", mfc="#888888", mec="white", label=s)
      for s, m in [("135M", "o"), ("1.7B", "s"), ("7B", "D")]]
leg1 = ax.legend(handles=h1, loc="lower left", fontsize=7, title="campaign", title_fontsize=7)
ax.add_artist(leg1)
model_handles, model_labels = ax.get_legend_handles_labels()
ax.legend(handles=h2 + model_handles, loc="upper right", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_r_of_T_raw_mu09.png"))
plt.close(fig)

# --- Fig 2: small multiples, every stratum -----------------------------
plot_strata = [(c, m) for (c, m) in sorted(strata.keys())
               if len(strata[(c, m)]["fit"]["T"].unique()) >= 2]
ncol = 4
nrow = int(math.ceil(len(plot_strata) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.9 * nrow),
                         sharey=False)
axes = np.atleast_2d(axes)
for i, (conv, mu) in enumerate(plot_strata):
    ax = axes[i // ncol][i % ncol]
    g = strata[(conv, mu)]["all"]
    for _, p in g.iterrows():
        col = PAL[0] if p["scale"] == "135M" else (PAL[1] if p["scale"] == "1.7B" else PAL[3])
        ax.errorbar(p["T"], p["r"],
                    yerr=[[max(p["r"] - p["r_ci_lo"], 0)], [max(p["r_ci_hi"] - p["r"], 0)]],
                    fmt="o", ms=4, mfc=col, mec="white", mew=0.6, ecolor=col,
                    elinewidth=0.9, alpha=0.55 if p["pair_type"] == "cross_campaign" else 1.0)
    Tmax = max(g["T"].max() * 1.6, 30)
    Tg = np.geomspace(1.5, Tmax, 200)
    ax.plot(Tg, [M_code(conv, mu, t) for t in Tg], "--", color="#666666", lw=1.2)
    best = pick_best(fits[(fits["convention"] == conv) & (fits["mu"] == mu)])
    if best is not None:
        yv = 2.0 ** model_curve(conv, mu, best["model"], json.loads(best["params"]), Tg)
        ax.plot(Tg, yv, "-", color="#222222", lw=1.5)
        ax.text(0.03, 0.05, f"best: {best['model']}\nchi2/dof={best['chi2_eff_per_dof']:.1f}"
                f"  p={best['p_eff']:.2g}",
                transform=ax.transAxes, fontsize=6.5, va="bottom")
    ax.axhline(1.0, color="#cccccc", lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(f"{conv}  mu={mu:g}", fontsize=8.5)
for j in range(len(plot_strata), nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
fig.suptitle("r(T) per stratum — dashed grey: code-true multiplier M(T); "
             "solid: best-fitting shape (highest chi2 GOF p, dof>=1)",
             fontsize=10, y=1.002)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_r_of_T_all_strata.png"), bbox_inches="tight")
plt.close(fig)

# --- Fig 3: S residuals in the G6 factorial ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
for ax, conv in zip(axes, ["nesterov_raw", "nesterov_corrected"]):
    g6 = pairs[(pairs["campaign"] == "G6") & (pairs["convention"] == conv)]
    Ts = sorted(g6["T"].unique())
    for k, t in enumerate(Ts):
        sub = g6[g6["T"] == t]
        base = sub["log2_r"].mean()
        ax.plot(np.log2(sub["S"] / 2560.0), sub["log2_r"] - base, "o-",
                color=PAL[k], ms=5, lw=1.2, mec="white", mew=0.6, label=f"T={t}")
    row = s_dep[s_dep["convention"] == conv].iloc[0]
    ax.set_title(f"{conv}: {row['bits_per_doubling_S']:+.3f} bits/doubling "
                 f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]", fontsize=8.5)
    ax.set_xlabel("log2(S / 2560)   (= log2 H shift at fixed T)")
    ax.axhline(0, color="#cccccc", lw=0.8)
ax = axes[0]
ax.set_ylabel("log2 r  minus per-T mean (bits)")
ax.legend(fontsize=7)
fig.suptitle("G6 factorial: S-dependence of the matched ratio after removing T structure", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_s_dependence.png"), bbox_inches="tight")
plt.close(fig)

print("\n[fits]")
show = fits[["convention", "mu", "model", "n", "dof", "chi2_eff", "chi2_eff_per_dof",
             "p_eff", "aicc", "rmse_bits", "params"]]
print(show.to_string())

# ----------------------------------------------------------------------
# Leave-one-campaign-out stability of the winning shapes (rich strata)
# ----------------------------------------------------------------------
loco_rows = []
for (conv, mu), model_name in [(("nesterov_raw", 0.9), "saturating_exp"),
                               (("nesterov_raw", 0.9), "sat_exp_to_one"),
                               (("nesterov_corrected", 0.9), "power_law"),
                               (("nesterov_corrected", 0.9), "exp_decay")]:
    st = strata[(conv, mu)]
    g_all = st["fit"]
    s_rep = st["s_rep"] if np.isfinite(st["s_rep"]) else S_REP_FALLBACK

    def _logM_cont(t, conv=conv, mu=mu):
        return math.log(M_code(conv, mu, t)) if conv in (
            "nesterov_raw", "heavy_ball") else 0.0
    globals()["_logM_cont"] = _logM_cont

    for drop in ["(none)"] + sorted(g_all["campaign"].unique()):
        g = g_all if drop == "(none)" else g_all[g_all["campaign"] != drop]
        if len(g["T"].unique()) < 3:
            continue
        T = g["T"].to_numpy(float)
        y = g["log2_r"].to_numpy() * LOG2
        sig = np.sqrt(np.maximum(g["se_log_r"].to_numpy(), 1e-6) ** 2 + s_rep ** 2)
        mvec = np.array([_logM_cont(t) for t in T])
        res = fit_model(model_name, T, y, sig, mvec)
        loco_rows.append(dict(convention=conv, mu=mu, model=model_name,
                              dropped=drop, n=len(g),
                              chi2_eff_per_dof=float(((res["resid"] / sig) ** 2).sum())
                              / max(len(g) - res["k"], 1),
                              **{f"p_{k}": round(v, 5) for k, v in res["params"].items()}))
loco = pd.DataFrame(loco_rows)
loco.to_csv(os.path.join(HERE, "loco_stability.csv"), index=False)
print("\n[LOCO]")
print(loco.to_string())

print("\n[cells] raw mu0.9 cell means:")
print(cells[(cells["convention"] == "nesterov_raw") & (cells["mu"] == 0.9)].to_string())
print("\n[cells] corrected mu0.9 cell means:")
print(cells[(cells["convention"] == "nesterov_corrected") & (cells["mu"] == 0.9)].to_string())
