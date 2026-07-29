#!/usr/bin/env python3
"""law-v2 zoo: competitive fitting of candidate outer tuning laws.

Fits a zoo of candidate laws for eta*(T, mu, S, scale, convention) against the
112 banked tuned optima in mech/law-unification/tuned_optima.csv, with a
leave-one-CAMPAIGN-out (LOCO) overfitting firewall.

Design rules (frozen before fitting):
  * Response is y = log2(eta*) in bits.  Equal-point objective (per
    mech/law-unification/INCLUSION.md the primary fit is equal-point least
    squares in log space; narrow/correlated/singleton intervals must not
    dominate).
  * Every model gets exactly one free intercept per model SCALE (135M, 1.7B,
    7B).  These are profiled in closed form.  NO per-campaign parameters are
    admitted anywhere; models with per-campaign free parameters are
    disqualified by construction.
  * Per-convention parameters are allowed (convention is a physical stratum),
    but are reported separately in the parameter budget.
  * LOCO fallback rules, fixed in advance and identical for every model:
      - if the held-out campaign's scale is absent from training (only G9B/7B),
        predict with the nearest smaller trained scale intercept (1.7B);
      - if a per-convention parameter is unidentified in training (only
        G12/heavy_ball), fall back to the nesterov_raw value (heavy-ball's
        finite-age factor is the raw factor shifted by one age unit).
    Both fallbacks are identical across models, so league ordering is not
    driven by them.
  * Primary metric: pooled held-out log2-eta RMSE in bits over all 112 points
    (each point predicted exactly once, by the fit that never saw its
    campaign); per-campaign RMSE reported.  Secondary: full-fit AIC/BIC and
    the frozen coverage/heterogeneity criteria of INCLUSION.md.

Reproducible: fixed seed, deterministic multistart.  Run from repo root:
    .venv/bin/python mech/law-v2/zoo/fit_zoo.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

SEED = 20260728
N_STARTS = 24
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "mech" / "law-unification" / "tuned_optima.csv"
OUT = Path(__file__).resolve().parent

LN2 = math.log(2.0)
SCALES = ["135M", "1.7B", "7B"]
CONVS = ["mu0", "nesterov_raw", "nesterov_corrected", "heavy_ball"]
# Fixed LOCO fallback maps (declared above).
SCALE_FALLBACK = {"7B": "1.7B", "1.7B": "135M", "135M": "135M"}
CONV_FALLBACK = {"heavy_ball": "nesterov_raw"}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    assert len(df) == 112
    df = df.copy()
    df["y"] = np.log2(df["eta_star"])
    df["se_bits"] = df["se_log_eta"] / LN2
    df["l2_one_minus_mu"] = np.log2(1.0 - df["mu"])
    # code-true finite-age multiplier M is provided in the CSV; recompute to
    # verify and to have it available at arbitrary (T, mu).
    M = np.array([finite_age_M(t, m, c)
                  for t, m, c in zip(df["T"], df["mu"], df["convention"])])
    assert np.allclose(M, df["M"], rtol=1e-6), "M mismatch vs ledger"
    return df


def finite_age_M(T: float, mu: float, conv: str) -> float:
    if mu == 0.0 or conv in ("mu0", "nesterov_corrected"):
        return 1.0
    if conv == "nesterov_raw":
        return (1.0 - mu ** (T + 1)) / (1.0 - mu)
    if conv == "heavy_ball":
        return (1.0 - mu ** T) / (1.0 - mu)
    raise ValueError(conv)


def accumulated_C(T: float, mu: float, conv: str) -> float:
    """Accumulated per-fragment displacement coefficient from
    lean-mechanism/LeanMechanism/FiniteHorizonOuter.lean (theory lane C1):
    the frozen-gradient optimum equalizes eta* x C across arms, so the
    kinematic factor in the law is T/C (not the terminal multiplier M)."""
    if mu == 0.0 or conv == "mu0":
        return float(T)
    if conv == "nesterov_raw":
        return T / (1 - mu) - mu ** 2 * (1 - mu ** T) / (1 - mu) ** 2
    if conv == "heavy_ball":
        return T / (1 - mu) - mu * (1 - mu ** T) / (1 - mu) ** 2
    if conv == "nesterov_corrected":
        return T / (1 - mu)
    raise ValueError(conv)


# --------------------------------------------------------------------------
# model framework
# --------------------------------------------------------------------------
# A model supplies g(df, theta) = predicted y minus the scale intercept.
# Scale intercepts are profiled: a[s] = mean(y - g) over training rows of s.

@dataclass
class Model:
    name: str
    desc: str
    theta_names: list[str]
    lo: np.ndarray
    hi: np.ndarray
    g: object  # callable(df, theta) -> np.ndarray of bits
    n_global: int = 0          # global shape parameters (theta)
    n_stratum: int = 3         # per-stratum params: 3 scale intercepts (+conv)
    stratum_note: str = "3 scale intercepts"
    theta_hat: np.ndarray | None = None
    disqualified: bool = False
    # optional callable(train_df, theta)->theta applying the declared
    # unseen-convention fallback (heavy_ball -> nesterov_raw) at predict time
    fallback: object = None
    # round 2: models whose scale intercepts enter the law nonlinearly (the
    # C3 interference term needs eta0[scale] inside an exponent) carry the
    # three intercepts as theta[0:3] = log2 eta0[135M, 1.7B, 7B] instead of
    # profiling them.  Same equal-point objective, same parameter counting.
    explicit_intercepts: bool = False
    # optional callable(train, rng)->theta0 giving data-informed multistart
    # points (used by explicit-intercept models so Nelder-Mead does not have
    # to discover the intercepts from random corners)
    make_start: object = None

    @property
    def k(self) -> int:
        return self.n_global + self.n_stratum


def pack(df: pd.DataFrame) -> dict:
    """Convert a dataframe slice to plain numpy arrays (optimizer hot path)."""
    d = dict(
        T=df["T"].to_numpy(dtype=float),
        mu=df["mu"].to_numpy(dtype=float),
        S=df["S"].to_numpy(dtype=float),
        M=df["M"].to_numpy(dtype=float),
        conv=df["convention"].to_numpy(),
        l1m=df["l2_one_minus_mu"].to_numpy(),
        y=df["y"].to_numpy(),
        scale=df["scale"].to_numpy(),
    )
    d["l2M"] = np.log2(d["M"])
    d["l2T"] = np.log2(d["T"])
    d["H"] = df["H"].to_numpy(dtype=float)
    d["C"] = np.array([accumulated_C(t, m, c) for t, m, c in
                       zip(d["T"], d["mu"], d["conv"])])
    d["l2kin"] = d["l2T"] - np.log2(d["C"])  # log2(T/C), theory C1 factor
    d["scale_idx"] = np.array([SCALES.index(s) for s in d["scale"]])
    d["scale_masks"] = {s: d["scale"] == s for s in SCALES
                        if (d["scale"] == s).any()}
    return d


def profile_intercepts(d: dict, gvals: np.ndarray) -> dict[str, float]:
    resid = d["y"] - gvals
    return {s: float(resid[m].mean()) for s, m in d["scale_masks"].items()}


def intercept_for(scale: str, a: dict[str, float]) -> float:
    s = scale
    while s not in a:
        nxt = SCALE_FALLBACK[s]
        if nxt == s:
            raise RuntimeError("no trained scale intercept at all")
        s = nxt
    return a[s]


def predict(model: Model, theta: np.ndarray, train: dict,
            test: dict) -> np.ndarray:
    if model.fallback is not None:
        theta = model.fallback(train, np.array(theta, dtype=float))
    if model.explicit_intercepts:
        return model.g(test, theta)
    a = profile_intercepts(train, model.g(train, theta))
    gt = model.g(test, theta)
    aa = np.array([intercept_for(s, a) for s in test["scale"]])
    return aa + gt


def fit(model: Model, train: dict, rng: np.random.Generator) -> np.ndarray:
    """Equal-point least squares with profiled intercepts, multistart."""
    y = train["y"]
    masks = list(train["scale_masks"].values())

    if model.explicit_intercepts:
        def sse(theta: np.ndarray) -> float:
            r = y - model.g(train, theta)
            return float(r @ r)
    else:
        def sse(theta: np.ndarray) -> float:
            g = model.g(train, theta)
            r = y - g
            tot = 0.0
            for m in masks:
                rm = r[m]
                rm = rm - rm.mean()
                tot += float(rm @ rm)
            return tot

    if len(model.theta_names) == 0:
        return np.zeros(0)

    best_x, best_f = None, np.inf
    if model.make_start is not None:
        starts = [model.make_start(train, rng) for _ in range(N_STARTS)]
    else:
        mid = (model.lo + model.hi) / 2.0
        starts = [mid]
        for _ in range(N_STARTS - 1):
            starts.append(model.lo
                          + rng.random(len(model.lo)) * (model.hi - model.lo))
    for x0 in starts:
        res = optimize.minimize(
            sse, x0, method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-11})
        x = np.clip(res.x, model.lo, model.hi)
        f = sse(x)
        if f < best_f:
            best_f, best_x = f, x
    # polish
    res = optimize.minimize(sse, best_x, method="Nelder-Mead",
                            options={"maxiter": 12000, "xatol": 1e-11,
                                     "fatol": 1e-13})
    x = np.clip(res.x, model.lo, model.hi)
    if sse(x) < best_f:
        best_x = x
    return best_x


# --------------------------------------------------------------------------
# candidate laws
# --------------------------------------------------------------------------
# Shorthand pulls used by every g().

def cols(d: dict):
    return d["T"], d["mu"], d["S"], d["M"], d["conv"], d["l1m"]


def conv_onehot(conv: np.ndarray) -> dict[str, np.ndarray]:
    return {c: (conv == c) for c in CONVS}


def softplus(x: np.ndarray, w: float) -> np.ndarray:
    return w * np.logaddexp(0.0, x / w)


def make_zoo() -> list[Model]:
    zoo: list[Model] = []

    # B0 -- FAILED BASELINE (frozen keystone law):
    #   eta* = eta0[scale] * (1-mu) * M * q**T
    def g_B0(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        (q,) = th
        return l1m + np.log2(M) + T * math.log2(q)
    zoo.append(Model("B0-baseline", "eta0[s]*(1-mu)*M*q^T (failed keystone)",
                     ["q"], np.array([0.85]), np.array([1.05]), g_B0,
                     n_global=1))

    # P1 -- multiplier times power law:  eta* = eta0[s]*(1-mu)*M*T^-alpha
    def g_P1(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        (alpha,) = th
        return l1m + np.log2(M) - alpha * np.log2(T)
    zoo.append(Model("P1-mult-powerlaw", "eta0[s]*(1-mu)*M*T^-a",
                     ["alpha"], np.array([-1.0]), np.array([3.0]), g_P1,
                     n_global=1))

    # Q1 -- multiplier times q^T with convention-specific q.
    def g_Q1(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        qm = dict(zip(CONVS, th))
        lq = np.array([math.log2(qm[c]) for c in conv])
        return l1m + np.log2(M) + T * lq
    def fb_Q1(train, th):
        if not (train["conv"] == "heavy_ball").any():
            th[3] = th[1]  # q_hb <- q_raw
        return th
    zoo.append(Model("Q1-conv-q", "eta0[s]*(1-mu)*M*q_conv^T (q per convention)",
                     [f"q_{c}" for c in CONVS],
                     np.array([0.85] * 4), np.array([1.05] * 4), g_Q1,
                     n_global=0, n_stratum=7,
                     stratum_note="3 scale intercepts + 4 convention q",
                     fallback=fb_Q1))

    # Q2 -- mu-dependent decay rate: log q(mu) = log q0 + kappa*mu
    def g_Q2(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        q0, kap = th
        return l1m + np.log2(M) + T * (math.log2(q0) + kap * mu)
    zoo.append(Model("Q2-mu-slope", "eta0[s]*(1-mu)*M*exp(T*(ln q0 + k*mu))",
                     ["q0", "kappa"], np.array([0.85, -0.2]),
                     np.array([1.05, 0.2]), g_Q2, n_global=2))

    # X1 -- two-regime crossover, mu-dependent crossover age tau(mu)=c/(1-mu):
    #   multiplier regime for T << tau, q-decay for T >> tau (smooth hinge).
    def g_X1(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        q, c = th
        tau = c / (1.0 - mu)
        return l1m + np.log2(M) + math.log2(q) * softplus(T - tau, 2.0)
    zoo.append(Model("X1-crossover-tau(mu)",
                     "multiplier regime -> q^(T-tau) beyond tau(mu)=c/(1-mu)",
                     ["q", "c"], np.array([0.7, 0.05]),
                     np.array([1.0, 20.0]), g_X1, n_global=2))

    # X2 -- two-regime crossover with constant tau (is mu-dependence needed?)
    def g_X2(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        q, tau = th
        return l1m + np.log2(M) + math.log2(q) * softplus(T - tau, 2.0)
    zoo.append(Model("X2-crossover-const-tau",
                     "multiplier regime -> q^(T-tau) beyond constant tau",
                     ["q", "tau"], np.array([0.7, 0.0]),
                     np.array([1.0, 80.0]), g_X2, n_global=2))

    # R1 -- rho-decay: effective multiplier with decaying alignment,
    #   M_eff = 1 + rho(T)*(M-1), rho(T) = rho0 * d^T, plus shared q^T.
    def g_R1(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        rho0, d, q = th
        Meff = 1.0 + rho0 * d ** T * (M - 1.0)
        return l1m + np.log2(Meff) + T * math.log2(q)
    zoo.append(Model("R1-rho-decay", "M_eff=1+rho0*d^T*(M-1); *q^T",
                     ["rho0", "d", "q"], np.array([0.01, 0.05, 0.85]),
                     np.array([64.0, 0.999, 1.05]), g_R1, n_global=3))

    # R2 -- rho-decay on the momentum clock: rho = rho0 * d^(T*(1-mu)).
    def g_R2(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        rho0, d, q = th
        Meff = 1.0 + rho0 * d ** (T * (1.0 - mu)) * (M - 1.0)
        return l1m + np.log2(Meff) + T * math.log2(q)
    zoo.append(Model("R2-rho-mu-clock", "M_eff=1+rho0*d^(T(1-mu))*(M-1); *q^T",
                     ["rho0", "d", "q"], np.array([0.01, 1e-4, 0.85]),
                     np.array([64.0, 0.999, 1.05]), g_R2, n_global=3))

    return zoo


# Own-design models are appended by add_own_designs() after residual
# inspection of the pre-registered eight above; see RESULTS.md for the
# motivating residual structure.

def add_own_designs(zoo: list[Model]) -> None:
    # H1 -- rho-decay + a separate decay rate for the bias-corrected arm.
    # Motivation: matched-cell ledger shows the corrected/mu0 tuned ratio
    # decays gently and log-linearly while raw needs the rho collapse.
    def g_H1(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        rho0, d, q, qc = th
        Meff = 1.0 + rho0 * d ** T * (M - 1.0)
        lq = np.where(conv == "nesterov_corrected", math.log2(qc),
                      math.log2(q))
        return l1m + np.log2(Meff) + T * lq
    zoo.append(Model("H1-rho+q_corr", "rho-decay M_eff; q shared, q_corr free",
                     ["rho0", "d", "q", "q_corr"],
                     np.array([0.01, 0.05, 0.85, 0.85]),
                     np.array([64.0, 0.999, 1.05, 1.05]), g_H1,
                     n_global=3, n_stratum=4,
                     stratum_note="3 scale intercepts + 1 corrected-arm q"))

    # H2 -- power-law transfer ratio: momentum arms sit a *power law in T*
    # below the mu0 curve (matched-cell medians are log-linear in log T for
    # both raw and corrected), instead of the multiplier M ever being right:
    #   eta*(mu0)   = eta0[s] * q^T
    #   eta*(conv)  = eta*(mu0) * (1-mu) * 2^{c_conv} * T^{-beta_conv}
    def g_H2(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        q, c_raw, b_raw, c_cor, b_cor, c_hb, b_hb = th
        base = T * math.log2(q)
        mom = (mu > 0)
        add = np.zeros(len(T))
        for cname, cc, bb in (("nesterov_raw", c_raw, b_raw),
                              ("nesterov_corrected", c_cor, b_cor),
                              ("heavy_ball", c_hb, b_hb)):
            m = (conv == cname) & mom
            add[m] = l1m[m] + cc - bb * np.log2(T[m])
        return base + add
    def fb_H(train, th):
        if not (train["conv"] == "heavy_ball").any():
            th[5], th[6] = th[1], th[2]  # (c,b)_hb <- (c,b)_raw
        return th
    zoo.append(Model("H2-powerlaw-ratio",
                     "mu0: q^T; momentum: *(1-mu)*2^c_conv*T^-beta_conv",
                     ["q", "c_raw", "b_raw", "c_cor", "b_cor", "c_hb", "b_hb"],
                     np.array([0.85, -4, -1, -4, -1, -4, -1]),
                     np.array([1.05, 6, 3, 6, 3, 6, 3]), g_H2,
                     n_global=1, n_stratum=9,
                     stratum_note="3 scale intercepts + (c,beta) per momentum "
                                  "convention", fallback=fb_H))

    # H3 -- H2 with mu entering the ratio through the momentum horizon:
    #   ratio = (1-mu) * 2^{c_conv} * (T*(1-mu))^{-beta_conv}
    # (one shared beta scale-law on the effective age in momentum horizons;
    #  tests whether the power-law ratio's clock is T or T(1-mu)).
    def g_H3(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        q, c_raw, b_raw, c_cor, b_cor, c_hb, b_hb = th
        base = T * math.log2(q)
        mom = (mu > 0)
        add = np.zeros(len(T))
        age = np.maximum(T * (1.0 - mu), 1e-9)
        for cname, cc, bb in (("nesterov_raw", c_raw, b_raw),
                              ("nesterov_corrected", c_cor, b_cor),
                              ("heavy_ball", c_hb, b_hb)):
            m = (conv == cname) & mom
            add[m] = l1m[m] + cc - bb * np.log2(age[m])
        return base + add
    zoo.append(Model("H3-powerlaw-muclock",
                     "mu0: q^T; momentum: *(1-mu)*2^c*(T(1-mu))^-beta",
                     ["q", "c_raw", "b_raw", "c_cor", "b_cor", "c_hb", "b_hb"],
                     np.array([0.85, -6, -1, -6, -1, -6, -1]),
                     np.array([1.05, 6, 3, 6, 3, 6, 3]), g_H3,
                     n_global=1, n_stratum=9,
                     stratum_note="3 scale intercepts + (c,beta) per momentum "
                                  "convention", fallback=fb_H))

    # H4 -- OWN DESIGN (from round-1 held-out residuals).  Motivating
    # structure: (a) even the mu0 arm's held-out residuals under any q^T model
    # are convex in log T (+0.7 bits at T=2, -1.0 at T=40, +0.4 at T=160),
    # i.e. the base clock is closer to a power law than an exponential;
    # (b) the rho-decay effective multiplier fixes the raw arm; (c) the
    # corrected arm needs a small extra age tilt.
    #   eta* = eta0[s]*(1-mu)*M_eff(T)*T^-alpha * T^-beta_c[corrected]
    #   M_eff = 1 + rho0*d^T*(M-1)
    def g_H4(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        rho0, d, alpha, beta_c = th
        Meff = 1.0 + rho0 * d ** T * (M - 1.0)
        l2T = np.log2(T)
        extra = np.where(conv == "nesterov_corrected", beta_c, 0.0)
        return l1m + np.log2(Meff) - (alpha + extra) * l2T
    zoo.append(Model("H4-rho-powerlaw",
                     "M_eff=1+rho0*d^T*(M-1); base T^-a; corrected extra "
                     "T^-beta_c",
                     ["rho0", "d", "alpha", "beta_c"],
                     np.array([0.01, 0.05, -0.5, -1.0]),
                     np.array([64.0, 0.999, 2.0, 2.0]), g_H4,
                     n_global=3, n_stratum=4,
                     stratum_note="3 scale intercepts + 1 corrected-arm tilt"))

    # H5 -- OWN DESIGN: H4 plus a local-work tilt S^sigma.  The G6 factorial
    # shows a small positive S trend (~+0.1 bits per doubling) that no other
    # model carries; sigma is a single global exponent, not a per-campaign
    # dial.
    def g_H5(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        rho0, d, alpha, beta_c, sigma = th
        Meff = 1.0 + rho0 * d ** T * (M - 1.0)
        l2T = np.log2(T)
        extra = np.where(conv == "nesterov_corrected", beta_c, 0.0)
        return (l1m + np.log2(Meff) - (alpha + extra) * l2T
                + sigma * np.log2(S / 2560.0))
    zoo.append(Model("H5-rho-powerlaw-S",
                     "H4 + global local-work tilt (S/2560)^sigma",
                     ["rho0", "d", "alpha", "beta_c", "sigma"],
                     np.array([0.01, 0.05, -0.5, -1.0, -1.0]),
                     np.array([64.0, 0.999, 2.0, 2.0, 1.0]), g_H5,
                     n_global=4, n_stratum=4,
                     stratum_note="3 scale intercepts + 1 corrected-arm tilt"))

    # H6 -- OWN DESIGN refinement: H4 with a saturating base,
    #   base(T) = log2(T^-alpha + f),
    # motivated by the held-out T=160 upturn (+0.7 bits in both the mu0 and
    # raw arms, seen in two independent campaigns TP-v1/TP-v2): the tuned
    # outer rate stops decaying at very large age instead of following a pure
    # power law.
    def g_H6(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        rho0, d, alpha, f, beta_c = th
        Meff = 1.0 + rho0 * d ** T * (M - 1.0)
        base = np.log2(T ** (-alpha) + f)
        extra = np.where(conv == "nesterov_corrected", beta_c, 0.0)
        return l1m + np.log2(Meff) + base - extra * np.log2(T)
    zoo.append(Model("H6-rho-powerlaw-floor",
                     "M_eff=1+rho0*d^T*(M-1); base (T^-a + f); corrected "
                     "extra T^-beta_c",
                     ["rho0", "d", "alpha", "f", "beta_c"],
                     np.array([0.01, 0.05, -0.5, 0.0, -1.0]),
                     np.array([64.0, 0.999, 2.5, 0.9, 2.0]), g_H6,
                     n_global=4, n_stratum=4,
                     stratum_note="3 scale intercepts + 1 corrected-arm tilt"))

    # H7 -- H6 + global local-work tilt (S/2560)^sigma.  Once the base-clock
    # misfit is removed by the floor, the held-out residuals of H6 show a
    # clear S trend (Spearman rho=0.39, p=2e-5) that was masked before; H7
    # tests whether one global exponent carries it across campaigns.
    def g_H7(df, th):
        T, mu, S, M, conv, l1m = cols(df)
        rho0, d, alpha, f, beta_c, sigma = th
        Meff = 1.0 + rho0 * d ** T * (M - 1.0)
        base = np.log2(T ** (-alpha) + f)
        extra = np.where(conv == "nesterov_corrected", beta_c, 0.0)
        return (l1m + np.log2(Meff) + base - extra * np.log2(T)
                + sigma * np.log2(S / 2560.0))
    zoo.append(Model("H7-rho-floor-S",
                     "H6 + global local-work tilt (S/2560)^sigma",
                     ["rho0", "d", "alpha", "f", "beta_c", "sigma"],
                     np.array([0.01, 0.05, -0.5, 0.0, -1.0, -1.0]),
                     np.array([64.0, 0.999, 2.5, 0.9, 2.0, 1.0]), g_H7,
                     n_global=5, n_stratum=4,
                     stratum_note="3 scale intercepts + 1 corrected-arm tilt"))


# --------------------------------------------------------------------------
# round 2: theory-lane candidates (mech/law-v2/theory/CANDIDATES.md, 0dde35e)
# --------------------------------------------------------------------------
# C1: replace the terminal multiplier (1-mu)M by the theorem-backed kinematic
# factor T/C_conv (accumulated displacement, FiniteHorizonOuter.lean).  Zero
# free parameters in the transform.  Scored with both clocks: q^T and the
# round-1 saturating power law (refit, not frozen at H7's values).
# C1+C3: times the interference factor
#     phi = exp(-beta * sqrt(H) * eta0[scale] * base(T) * (C - T)),
# ONE global beta, refit inside every LOCO fold (no leakage of theory beta).

LOG2E = 1.0 / LN2


def add_theory_candidates(zoo: list[Model]) -> None:
    # ---- C1 pure, q^T clock (theory keystone form, params same as B0) ----
    def g_C1q(df, th):
        (q,) = th
        return df["l2kin"] + df["T"] * math.log2(q)
    zoo.append(Model("C1q-pure", "eta0[s]*(T/C_conv)*q^T (theorem kinematics)",
                     ["q"], np.array([0.85]), np.array([1.05]), g_C1q,
                     n_global=1))

    # ---- C1 pure, saturating power-law clock (shape refit) ----
    def g_C1sat(df, th):
        alpha, f = th
        return df["l2kin"] + np.log2(df["T"] ** (-alpha) + f)
    zoo.append(Model("C1sat-pure", "eta0[s]*(T/C_conv)*(T^-a+f)",
                     ["alpha", "f"], np.array([-0.5, 0.0]),
                     np.array([2.5, 0.9]), g_C1sat, n_global=2))

    # ---- helpers for explicit-intercept C3 models ----
    def fb_expl(train, th):
        sm = train["scale_masks"]
        if "1.7B" not in sm:
            th[1] = th[0]
        if "7B" not in sm:
            th[2] = th[1]  # declared fallback: unseen 7B -> trained 1.7B
        return th

    def smart_start(shape_sampler, g_shape):
        """Start with random shape params, intercepts from one profiling
        pass at beta=0, small random beta."""
        def _mk(train, rng):
            shape = shape_sampler(rng)
            gs = g_shape(train, shape)
            resid = train["y"] - gs
            a = []
            prev = -3.0
            for s in SCALES:
                m = train["scale_masks"].get(s)
                a.append(float(resid[m].mean()) if m is not None else prev)
                prev = a[-1]
            beta = rng.uniform(0.0, 0.02)
            return np.array(a + list(shape) + [beta])
        return _mk

    # ---- C1 + C3, q^T clock ----
    def g_C1qC3(df, th):
        a = np.asarray(th[0:3])[df["scale_idx"]]
        q, beta = th[3], th[4]
        base = q ** df["T"]
        phi_l2 = (-beta * np.sqrt(df["H"]) * 2.0 ** a * base
                  * (df["C"] - df["T"])) * LOG2E
        return a + df["l2kin"] + df["T"] * math.log2(q) + phi_l2
    zoo.append(Model(
        "C1q+C3", "eta0[s]*(T/C)*q^T * exp(-beta sqrt(H) eta0 q^T (C-T))",
        ["a_135M", "a_1.7B", "a_7B", "q", "beta"],
        np.array([-9.0, -9.0, -9.0, 0.85, 0.0]),
        np.array([-1.0, -1.0, -1.0, 1.05, 0.1]),
        g_C1qC3, n_global=2, n_stratum=3,
        stratum_note="3 scale intercepts (explicit)",
        fallback=fb_expl, explicit_intercepts=True,
        make_start=smart_start(
            lambda rng: [rng.uniform(0.92, 1.01)],
            lambda tr, sh: tr["l2kin"] + tr["T"] * math.log2(sh[0]))))

    # ---- C1 + C3, saturating clock ----
    def g_C1satC3(df, th):
        a = np.asarray(th[0:3])[df["scale_idx"]]
        alpha, f, beta = th[3], th[4], th[5]
        base = df["T"] ** (-alpha) + f
        phi_l2 = (-beta * np.sqrt(df["H"]) * 2.0 ** a * base
                  * (df["C"] - df["T"])) * LOG2E
        return a + df["l2kin"] + np.log2(base) + phi_l2
    zoo.append(Model(
        "C1sat+C3", "eta0[s]*(T/C)*(T^-a+f) * exp(-beta sqrt(H) eta0 base "
                    "(C-T))",
        ["a_135M", "a_1.7B", "a_7B", "alpha", "f", "beta"],
        np.array([-9.0, -9.0, -9.0, -0.5, 0.0, 0.0]),
        np.array([-1.0, -1.0, -1.0, 2.5, 0.9, 0.1]),
        g_C1satC3, n_global=3, n_stratum=3,
        stratum_note="3 scale intercepts (explicit)",
        fallback=fb_expl, explicit_intercepts=True,
        make_start=smart_start(
            lambda rng: [rng.uniform(0.2, 2.0), rng.uniform(0.0, 0.4)],
            lambda tr, sh: tr["l2kin"]
            + np.log2(tr["T"] ** (-sh[0]) + sh[1]))))

    # ---- H7 with its fitted alignment term REPLACED by T/C ----
    def g_C1H7(df, th):
        alpha, f, sigma, beta_c = th
        extra = np.where(df["conv"] == "nesterov_corrected", beta_c, 0.0)
        return (df["l2kin"] + np.log2(df["T"] ** (-alpha) + f)
                - extra * df["l2T"] + sigma * np.log2(df["S"] / 2560.0))
    zoo.append(Model("C1sat-H7kin",
                     "H7 with (1-mu)M_eff replaced by theorem T/C (rho0,d "
                     "dropped)",
                     ["alpha", "f", "sigma", "beta_c"],
                     np.array([-0.5, 0.0, -1.0, -1.0]),
                     np.array([2.5, 0.9, 1.0, 2.0]), g_C1H7,
                     n_global=3, n_stratum=4,
                     stratum_note="3 scale intercepts + 1 corrected-arm "
                                  "tilt"))

    # ---- subsumption test: H7's alignment factor kept ON TOP of T/C ----
    # If the theorem kinematics already carries the alignment decay, the
    # refit rho0 should go to ~0 and LOCO should not improve.
    def g_C1H7a(df, th):
        rho0, d, alpha, f, sigma, beta_c = th
        align = 1.0 + rho0 * d ** df["T"] * (df["M"] - 1.0)
        extra = np.where(df["conv"] == "nesterov_corrected", beta_c, 0.0)
        return (df["l2kin"] + np.log2(align)
                + np.log2(df["T"] ** (-alpha) + f)
                - extra * df["l2T"] + sigma * np.log2(df["S"] / 2560.0))
    zoo.append(Model("C1sat-H7kin+align",
                     "C1sat-H7kin with H7's rho0*d^T*(M-1) alignment kept on "
                     "top of T/C",
                     ["rho0", "d", "alpha", "f", "sigma", "beta_c"],
                     np.array([0.0, 0.05, -0.5, 0.0, -1.0, -1.0]),
                     np.array([64.0, 0.999, 2.5, 0.9, 1.0, 2.0]), g_C1H7a,
                     n_global=5, n_stratum=4,
                     stratum_note="3 scale intercepts + 1 corrected-arm "
                                  "tilt"))


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def loco(model: Model, df: pd.DataFrame, rng_seed: int):
    campaigns = sorted(df["campaign"].unique())
    resid = np.full(len(df), np.nan)
    per_c = {}
    for i, camp in enumerate(campaigns):
        rng = np.random.default_rng(rng_seed + 7919 * i)
        tr = df[df["campaign"] != camp]
        te = df[df["campaign"] == camp]
        trp, tep = pack(tr), pack(te)
        th = fit(model, trp, rng)
        pred = predict(model, th, trp, tep)
        r = te["y"].to_numpy() - pred
        resid[te.index.to_numpy()] = r
        per_c[camp] = float(np.sqrt(np.mean(r ** 2)))
    assert not np.isnan(resid).any()
    pooled = float(np.sqrt(np.mean(resid ** 2)))
    return pooled, per_c, resid


def full_fit_stats(model: Model, df: pd.DataFrame, rng_seed: int):
    rng = np.random.default_rng(rng_seed)
    dp = pack(df)
    th = fit(model, dp, rng)
    model.theta_hat = th
    pred = predict(model, th, dp, dp)
    r = df["y"].to_numpy() - pred
    n = len(df)
    sse = float(r @ r)  # bits^2
    k = model.k
    aic = n * math.log(sse / n) + 2 * (k + 1)
    bic = n * math.log(sse / n) + (k + 1) * math.log(n)
    rmse = math.sqrt(sse / n)
    # frozen criteria (INCLUSION.md): heterogeneity + coverage on
    # noise-eligible points.
    el = df["noise_eligible"].astype(bool).to_numpy()
    z = (r[el] * LN2) / df["se_log_eta"].to_numpy()[el]
    chi2 = float(z @ z)
    dof = int(el.sum() - k)
    het_p = float(stats.chi2.sf(chi2, dof)) if dof > 0 else float("nan")
    eta_pred = 2.0 ** pred
    cover = ((df["eta_ci_low"].to_numpy() <= eta_pred)
             & (eta_pred <= df["eta_ci_high"].to_numpy()))
    coverage = float(cover[el].mean())
    return dict(theta=[float(x) for x in th], rmse_full=rmse, aic=aic,
                bic=bic, chi2=chi2, dof=dof, het_p=het_p, coverage=coverage,
                pred=pred, resid=r)


def jackknife_delta(res_a: np.ndarray, res_b: np.ndarray,
                    df: pd.DataFrame):
    """Campaign-level jackknife SE of pooled ΔRMSE (a minus b)."""
    campaigns = sorted(df["campaign"].unique())
    full = (np.sqrt(np.mean(res_a ** 2)) - np.sqrt(np.mean(res_b ** 2)))
    vals = []
    for camp in campaigns:
        m = (df["campaign"] != camp).to_numpy()
        vals.append(np.sqrt(np.mean(res_a[m] ** 2))
                    - np.sqrt(np.mean(res_b[m] ** 2)))
    vals = np.array(vals)
    g = len(campaigns)
    se = math.sqrt((g - 1) / g * np.sum((vals - vals.mean()) ** 2))
    return float(full), float(se)


# --------------------------------------------------------------------------
# figures (winner = H7 family)
# --------------------------------------------------------------------------

CONV_COLOR = {  # validated categorical palette (dataviz six-checks, light)
    "mu0": "#2a78d6",
    "nesterov_raw": "#eb6834",
    "nesterov_corrected": "#1baf7a",
    "heavy_ball": "#eda100",
}
CONV_LABEL = {
    "mu0": "mu = 0",
    "nesterov_raw": "Nesterov raw",
    "nesterov_corrected": "Nesterov corrected",
    "heavy_ball": "heavy ball",
}
SCALE_MARKER = {"135M": "o", "1.7B": "s", "7B": "^"}


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.grid(True, color="#ececea", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#5f5e56", labelsize=9)


def winner_strip(dp: dict, model: Model):
    """Return (strip, curve_fn, subtitle): y - strip should fall on
    curve_fn(T) for the given fitted model."""
    th = model.theta_hat
    name = model.name
    T = dp["T"]
    if model.explicit_intercepts:
        aa = np.asarray(th[0:3])[dp["scale_idx"]]
    else:
        a = profile_intercepts(dp, model.g(dp, th))
        aa = np.array([a[s] for s in dp["scale"]])
    if name == "H7-rho-floor-S":
        rho0, d, alpha, f, beta_c, sigma = th
        Meff = 1.0 + rho0 * d ** T * (dp["M"] - 1.0)
        extra = np.where(dp["conv"] == "nesterov_corrected", beta_c, 0.0)
        strip = (aa + dp["l1m"] + np.log2(Meff) - extra * np.log2(T)
                 + sigma * np.log2(dp["S"] / 2560.0))
        return (strip, lambda t: np.log2(t ** (-alpha) + f),
                "line = log2(T^-alpha + f), alpha=%.3f, f=%.3f" % (alpha, f))
    if name == "C1sat-H7kin":
        alpha, f, sigma, beta_c = th
        extra = np.where(dp["conv"] == "nesterov_corrected", beta_c, 0.0)
        strip = (aa + dp["l2kin"] - extra * dp["l2T"]
                 + sigma * np.log2(dp["S"] / 2560.0))
        return (strip, lambda t: np.log2(t ** (-alpha) + f),
                "kinematics = theorem T/C; line = log2(T^-alpha + f), "
                "alpha=%.3f, f=%.3f" % (alpha, f))
    if name == "C1sat-H7kin+align":
        rho0, d, alpha, f, sigma, beta_c = th
        align = 1.0 + rho0 * d ** T * (dp["M"] - 1.0)
        extra = np.where(dp["conv"] == "nesterov_corrected", beta_c, 0.0)
        strip = (aa + dp["l2kin"] + np.log2(align) - extra * dp["l2T"]
                 + sigma * np.log2(dp["S"] / 2560.0))
        return (strip, lambda t: np.log2(t ** (-alpha) + f),
                "T/C + residual alignment; line = log2(T^-alpha + f), "
                "alpha=%.3f, f=%.3f" % (alpha, f))
    if name == "C1sat+C3":
        alpha, f, beta = th[3], th[4], th[5]
        base = T ** (-alpha) + f
        phi = (-beta * np.sqrt(dp["H"]) * 2.0 ** aa * base
               * (dp["C"] - T)) * LOG2E
        strip = aa + dp["l2kin"] + phi
        return (strip, lambda t: np.log2(t ** (-alpha) + f),
                "T/C * C3 interference; line = log2(T^-alpha + f), "
                "alpha=%.3f, f=%.3f" % (alpha, f))
    if name == "C1q+C3":
        q, beta = th[3], th[4]
        phi = (-beta * np.sqrt(dp["H"]) * 2.0 ** aa * q ** T
               * (dp["C"] - T)) * LOG2E
        strip = aa + dp["l2kin"] + phi
        return (strip, lambda t: t * math.log2(q),
                "T/C * C3 interference; line = T log2 q, q=%.4f" % q)
    if name in ("C1q-pure", "C1sat-pure"):
        strip = aa + dp["l2kin"]
        if name == "C1q-pure":
            q = th[0]
            return (strip, lambda t: t * math.log2(q),
                    "kinematics = theorem T/C; line = T log2 q, q=%.4f" % q)
        alpha, f = th
        return (strip, lambda t: np.log2(t ** (-alpha) + f),
                "kinematics = theorem T/C; line = log2(T^-alpha + f), "
                "alpha=%.3f, f=%.3f" % (alpha, f))
    return None


def make_collapse_figure(df: pd.DataFrame, model: Model,
                         fname: str = "collapse_winner"):
    """Winner collapse: strip everything except the base age curve; every
    convention/scale/S cell should land on the common clock curve."""
    dp = pack(df)
    spec = winner_strip(dp, model)
    if spec is None:
        return False
    strip, curve, subtitle = spec
    T = dp["T"]
    ynorm = dp["y"] - strip

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=200)
    tt = np.geomspace(1.8, 190, 300)
    ax.plot(tt, curve(tt), color="#5f5e56", lw=2.0, zorder=2, label=None)
    rng = np.random.default_rng(SEED)
    for conv in CONVS:
        m = dp["conv"] == conv
        if not m.any():
            continue
        for s, mk in SCALE_MARKER.items():
            ms = m & (dp["scale"] == s)
            if not ms.any():
                continue
            jit = T[ms] * np.exp(rng.normal(0, 0.03, ms.sum()))
            ax.scatter(jit, ynorm[ms], s=26, marker=mk,
                       facecolor=CONV_COLOR[conv], edgecolor="white",
                       linewidth=0.6, zorder=3,
                       label=CONV_LABEL[conv] if s == "135M" or conv == "mu0"
                       else None)
    # dedupe legend
    h, l = ax.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for hi, li in zip(h, l):
        if li not in seen:
            seen.add(li)
            hh.append(hi)
            ll.append(li)
    leg1 = ax.legend(hh, ll, frameon=False, fontsize=8, loc="lower left")
    ax.add_artist(leg1)
    smh = [plt.Line2D([], [], color="#5f5e56", marker=mk, ls="none",
                      markersize=6) for mk in SCALE_MARKER.values()]
    ax.legend(smh, list(SCALE_MARKER), frameon=False, fontsize=8,
              loc="upper right", title="scale", title_fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 40, 160])
    ax.set_xticklabels([2, 5, 10, 20, 40, 160])
    ax.set_xlabel("outer age T (rounds)", fontsize=10, color="#33322e")
    ax.set_ylabel("normalized tuned rate (bits)", fontsize=10,
                  color="#33322e")
    ax.set_title(
        "%s collapse: log2 eta* minus scale/kinematic/convention/S terms\n%s"
        % (model.name, subtitle),
        fontsize=10, color="#33322e", loc="left")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.png")
    fig.savefig(OUT / f"{fname}.pdf")
    plt.close(fig)
    return True


def make_residual_figure(df: pd.DataFrame, resid: np.ndarray, winner: str,
                         fname: str = "residuals_holdout"):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=200, sharey=True)
    rng = np.random.default_rng(SEED + 1)
    conv = df["convention"].to_numpy()

    ax = axes[0]
    T = df["T"].to_numpy(dtype=float)
    for c in CONVS:
        m = conv == c
        jit = T[m] * np.exp(rng.normal(0, 0.04, m.sum()))
        ax.scatter(jit, resid[m], s=20, color=CONV_COLOR[c],
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 40, 160])
    ax.set_xticklabels([2, 5, 10, 20, 40, 160])
    ax.set_xlabel("T (rounds)", fontsize=9, color="#33322e")
    ax.set_ylabel("held-out residual (bits)", fontsize=9, color="#33322e")

    ax = axes[1]
    mu = df["mu"].to_numpy(dtype=float)
    for c in CONVS:
        m = conv == c
        jit = mu[m] + rng.normal(0, 0.008, m.sum())
        ax.scatter(jit, resid[m], s=20, color=CONV_COLOR[c],
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlabel("mu", fontsize=9, color="#33322e")

    ax = axes[2]
    for i, c in enumerate(CONVS):
        m = conv == c
        x = i + rng.normal(0, 0.06, m.sum())
        ax.scatter(x, resid[m], s=20, color=CONV_COLOR[c],
                   edgecolor="white", linewidth=0.5, zorder=3)
        med = np.median(resid[m])
        ax.plot([i - 0.25, i + 0.25], [med, med], color="#33322e", lw=1.6,
                zorder=4)
    ax.set_xticks(range(len(CONVS)))
    ax.set_xticklabels(["mu=0", "raw", "corrected", "heavy\nball"],
                       fontsize=8)
    ax.set_xlabel("convention (bar = median)", fontsize=9, color="#33322e")

    for ax in axes:
        ax.axhline(0, color="#8a8878", lw=1.0, zorder=1)
        style_ax(ax)
    fig.suptitle(f"{winner}: leave-one-campaign-out residuals",
                 fontsize=10, color="#33322e", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / f"{fname}.png")
    fig.savefig(OUT / f"{fname}.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    df = load()
    zoo = make_zoo()
    add_own_designs(zoo)
    add_theory_candidates(zoo)

    rows = []
    loco_resids = {}
    per_campaign = {}
    for m in zoo:
        st = full_fit_stats(m, df, SEED)
        pooled, per_c, resid = loco(m, df, SEED)
        loco_resids[m.name] = resid
        per_campaign[m.name] = per_c
        rows.append(dict(model=m.name, desc=m.desc,
                         n_global=m.n_global, n_stratum=m.n_stratum,
                         stratum_note=m.stratum_note, k=m.k,
                         theta=dict(zip(m.theta_names, st["theta"])),
                         rmse_full_bits=st["rmse_full"],
                         rmse_loco_bits=pooled,
                         aic=st["aic"], bic=st["bic"],
                         het_chi2=st["chi2"], het_dof=st["dof"],
                         het_p=st["het_p"], coverage=st["coverage"],
                         per_campaign=per_c))
        print(f"{m.name:26s} LOCO={pooled:6.3f}  full={st['rmse_full']:6.3f} "
              f"AIC={st['aic']:8.1f} BIC={st['bic']:8.1f} "
              f"cov={st['coverage']:.1%} k={m.k}")

    lg = pd.DataFrame(rows).sort_values("rmse_loco_bits")
    base_r = loco_resids["B0-baseline"]
    deltas = {}
    for m in zoo:
        d, se = jackknife_delta(loco_resids[m.name], base_r, df)
        wins = sum(per_campaign[m.name][c] < per_campaign["B0-baseline"][c]
                   for c in per_campaign[m.name])
        deltas[m.name] = dict(delta_vs_baseline=d, jackknife_se=se,
                              campaigns_improved=int(wins))
    lg["delta_vs_B0"] = [deltas[n]["delta_vs_baseline"] for n in lg["model"]]
    lg["delta_jk_se"] = [deltas[n]["jackknife_se"] for n in lg["model"]]
    lg["campaigns_improved"] = [deltas[n]["campaigns_improved"]
                                for n in lg["model"]]

    lg.to_csv(OUT / "league.csv", index=False)
    with open(OUT / "loco_residuals.json", "w") as f:
        json.dump({k: list(map(float, v)) for k, v in loco_resids.items()},
                  f, indent=1)
    with open(OUT / "fit_details.json", "w") as f:
        json.dump(rows, f, indent=1, default=str)

    # residual-structure tests (held-out residuals) for every model
    def structure_tests(r: np.ndarray) -> dict:
        tests = {}
        tests["spearman_T"] = stats.spearmanr(df["T"], r)._asdict()
        tests["spearman_mu"] = stats.spearmanr(df["mu"], r)._asdict()
        tests["spearman_S"] = stats.spearmanr(df["S"], r)._asdict()
        groups = [r[(df["convention"] == c).to_numpy()] for c in CONVS
                  if (df["convention"] == c).any()]
        kw = stats.kruskal(*groups)
        tests["kruskal_convention"] = dict(statistic=float(kw.statistic),
                                           pvalue=float(kw.pvalue))
        groups = [r[(df["scale"] == s).to_numpy()] for s in SCALES
                  if (df["scale"] == s).any()]
        kw = stats.kruskal(*groups)
        tests["kruskal_scale"] = dict(statistic=float(kw.statistic),
                                      pvalue=float(kw.pvalue))
        return tests

    all_tests = {m.name: structure_tests(loco_resids[m.name]) for m in zoo}
    winner = lg.iloc[0]["model"]
    with open(OUT / "winner_residual_tests.json", "w") as f:
        json.dump({"winner": winner, "tests": all_tests[winner]}, f,
                  indent=1, default=float)
    with open(OUT / "residual_tests_all.json", "w") as f:
        json.dump(all_tests, f, indent=1, default=float)
    print("\nwinner:", winner)
    for k, v in all_tests[winner].items():
        print(f"  {k}: {v}")

    # per-campaign held-out RMSE matrix
    pc = pd.DataFrame(per_campaign).T
    pc.index.name = "model"
    pc.round(4).to_csv(OUT / "loco_per_campaign.csv")

    # figures: winner, plus the best theory C-variant if it is not the winner
    by_name = {m.name: m for m in zoo}
    make_collapse_figure(df, by_name[winner])
    make_residual_figure(df, loco_resids[winner], winner)
    cbest = lg[lg["model"].str.startswith("C1")].iloc[0]["model"]
    if cbest != winner:
        make_collapse_figure(df, by_name[cbest], fname="collapse_best_C")
        make_residual_figure(df, loco_resids[cbest], cbest,
                             fname="residuals_holdout_best_C")
        print("best C-variant:", cbest)
        for k, v in all_tests[cbest].items():
            print(f"  {k}: {v}")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
