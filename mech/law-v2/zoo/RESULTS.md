# law-v2 zoo: competitive fits for the outer tuning law eta*(T, mu, S, scale, convention)

Data: all 112 accepted banked tuned optima in
`mech/law-unification/tuned_optima.csv` (12 campaigns, scales 135M/1.7B/7B,
conventions mu0 / nesterov_raw / nesterov_corrected / heavy_ball).
Response is `log2(eta*)` in bits; equal-point objective, per the frozen
primary-fit rule in `mech/law-unification/INCLUSION.md`.

Reproduce: `.venv/bin/python mech/law-v2/zoo/fit_zoo.py` (seed 20260728,
deterministic multistart Nelder-Mead; ~100 s).  Outputs: `league.csv`,
`loco_per_campaign.csv`, `loco_residuals.json`, `fit_details.json`,
`winner_residual_tests.json`, `collapse_winner.{png,pdf}`,
`residuals_holdout.{png,pdf}`.

## Protocol (fixed before scoring)

* **Primary metric: leave-one-CAMPAIGN-out (LOCO) cross-validation** over the
  12 campaigns.  Pooled held-out RMSE of log2(eta*) in bits (each point
  predicted exactly once by a fit that never saw its campaign), plus
  per-campaign RMSE.
* **Parameter budget.** Every model gets exactly 3 per-scale intercepts
  (135M/1.7B/7B), profiled in closed form.  Per-convention parameters are
  allowed and counted as stratum parameters.  **No model has any per-campaign
  parameter; any such model is disqualified by construction.**
* **LOCO fallbacks**, identical for every model, declared in advance:
  7B intercept unseen (G9B fold only) -> use the trained 1.7B intercept;
  heavy_ball parameters unseen (G12 fold only) -> use the nesterov_raw values.
  Because they are identical across models, they add the same irreducible
  penalty to every row of the league and cannot reorder it.
* **Win margin (declared):** the winner must beat the failed baseline B0 by at
  least 0.20 bits pooled LOCO RMSE, with Delta > 3x its campaign-level
  jackknife SE, improving a majority of campaigns.  Justification: the largest
  single-campaign leverage on B0's pooled LOCO RMSE is 0.034 bits (DISAMBIG),
  so 0.20 bits (~6x the maximum leverage) cannot be manufactured by one lucky
  campaign, and 3x the jackknife SE gives strong campaign-resampling
  significance for the paired difference.

## League table (sorted by held-out RMSE; bits of log2 eta*)

| model | form | k (global+stratum) | LOCO RMSE | full RMSE | AIC | BIC | Delta vs B0 (jk SE) | campaigns improved | 95% CI coverage | het p |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| **H7-rho-floor-S** (winner) | M_eff=1+rho0 d^T (M-1); base (T^-a+f); corrected tilt T^-b_c; (S/2560)^sigma | 9 (5+4) | **0.363** | 0.317 | -237.2 | -210.0 | **-1.010 (0.098)** | 10/12 | 10.4% | ~0 |
| H6-rho-powerlaw-floor | H7 without the S tilt | 8 (4+4) | 0.370 | 0.331 | -229.7 | -205.2 | -1.003 (0.105) | 10/12 | 9.4% | ~0 |
| H4-rho-powerlaw | M_eff rho-decay; pure power-law base T^-a | 7 (3+4) | 0.429 | 0.384 | -198.1 | -176.4 | -0.944 (0.088) | 10/12 | 14.2% | ~0 |
| H5-rho-powerlaw-S | H4 + S tilt | 8 (4+4) | 0.441 | 0.384 | -196.6 | -172.2 | -0.932 (0.084) | 10/12 | 12.3% | ~0 |
| H1-rho+q_corr | rho-decay M_eff; q^T base; separate q_corr | 7 (3+4) | 0.535 | 0.503 | -137.9 | -116.2 | -0.838 (0.059) | 11/12 | 6.6% | ~0 |
| R1-rho-decay | M_eff=1+rho0 d^T (M-1); shared q^T | 6 (3+3) | 0.607 | 0.576 | -109.7 | -90.6 | -0.767 (0.062) | 11/12 | 11.3% | ~0 |
| R2-rho-mu-clock | rho decays on T(1-mu) clock | 6 (3+3) | 0.635 | 0.598 | -101.3 | -82.2 | -0.739 (0.058) | 11/12 | 6.6% | ~0 |
| H2-powerlaw-ratio | mu0: q^T; momentum arms: (1-mu) 2^c_conv T^-b_conv | 10 (1+9) | 0.643 | 0.571 | -103.7 | -73.8 | -0.730 (0.058) | 12/12 | 10.4% | ~0 |
| H3-powerlaw-muclock | H2 on T(1-mu) clock | 10 (1+9) | 0.715 | 0.578 | -100.6 | -70.7 | -0.658 (0.088) | 11/12 | 5.7% | ~0 |
| P1-mult-powerlaw | eta0[s] (1-mu) M T^-a | 4 (1+3) | 1.054 | 0.998 | 9.5 | 23.1 | -0.319 (0.056) | 10/12 | 0.9% | ~0 |
| Q2-mu-slope | q(mu): ln q0 + kappa mu | 5 (2+3) | 1.289 | 1.196 | 52.1 | 68.4 | -0.084 (0.021) | 10/12 | 0.9% | ~0 |
| **B0-baseline** (failed keystone) | eta0[s] (1-mu) M q^T | 4 (1+3) | 1.373 | 1.310 | 70.5 | 84.1 | 0 | -- | 1.9% | ~0 |
| X2-crossover-const-tau | multiplier regime -> q^(T-tau), const tau | 5 (2+3) | 1.375 | 1.313 | 73.0 | 89.3 | +0.002 (0.001) | 2/12 | 0.9% | ~0 |
| X1-crossover-tau(mu) | crossover at tau(mu)=c/(1-mu) | 5 (2+3) | 1.379 | 1.315 | 73.4 | 89.7 | +0.006 (0.005) | 4/12 | 1.9% | ~0 |
| Q1-conv-q | q^T with q per convention | 7 (0+7) | 1.384 | 1.209 | 58.5 | 80.3 | +0.011 (0.105) | 10/12 | 0.0% | ~0 |

Firewall notes.  Q1 *improves the full fit* (1.209 vs 1.310) but is *worse
held-out* than the baseline: per-convention decay rates on the wrong
multiplier structure are pure overfit (its heavy-ball q=0.902 comes from a
single 3-point campaign).  The crossover family (X1/X2) collapses onto the
baseline (fitted tau -> 0): the data contain no "multiplier regime" -- the
effective multiplier only ever shrinks with age, it never holds.

## The winner

```
eta*(T, mu, S, scale, conv) =
    eta0[scale] * (1-mu) * M_eff(T, mu, conv) * (T^-alpha + f)
                * T^-beta_c[conv=corrected] * (S/2560)^sigma
    M_eff = 1 + rho0 * d^T * (M - 1),   M = the code-true finite-age factor

    rho0 = 4.066   d = 0.657   alpha = 1.256   f = 0.155
    beta_c = 0.150   sigma = 0.105
    eta0: 135M 0.1374, 1.7B 0.0409, 7B 0.0263   (full-fit intercepts)
```

Reading: the raw/heavy-ball finite-age multiplier M is never realized at
tuning time; its *alignment* decays exponentially with a ~1.6-round half-life
(d=0.657), so by T>~10 the tuned momentum optimum sits at plain (1-mu) times
the mu0 optimum.  The mu0 base curve is not an exponential q^T but a
saturating power law: T^-1.26 with a floor f=0.155 (the tuned rate stops
decaying around T~40; the T=160 optima sit at the floor).  The corrected arm
carries a small extra age tilt (T^-0.15), and there is a small positive
local-work exponent (S/2560)^0.105.

* Per-campaign held-out RMSE (bits), B0 -> H7:
  DISAMBIG 0.49->0.13, G12 1.80->0.52, G4C 1.63->0.41, G5B 1.18->1.32,
  G6 1.33->0.23, G8 1.35->0.40, G9A 1.11->0.12, G9B 0.49->0.64,
  TP-pilot 1.17->0.41, TP-v1 1.72->0.15, TP-v2 1.63->0.15, TP-v3 1.35->0.19.
* **Margin check:** Delta = -1.010 bits = 10.3x the jackknife SE (0.098),
  improving 10/12 campaigns -- clears the declared bar (0.20 bits, 3x SE,
  majority) by an order of magnitude.
* The two campaigns H7 does *not* improve are exactly the two structurally
  odd ones: G5B (SNOO; sits ~1.3 bits below every no-campaign-dial model) and
  G9B (7B, scored through the declared unseen-scale intercept fallback that
  penalizes all models identically).
* **Held-out residual structure (plots: `residuals_holdout_H7.png`;
  the unsuffixed `residuals_holdout.png` / `collapse_winner.png` now show the
  round-2 league winner, and `collapse_H7.png` keeps the H7 collapse):** no
  remaining structure at the 0.05 level -- Spearman vs T rho=0.077 (p=0.42),
  vs mu rho=0.121 (p=0.20), vs S rho=-0.092 (p=0.34); Kruskal-Wallis across
  conventions p=0.25; across scales p=0.078.  For contrast, H6 (no S term)
  leaves Spearman-vs-S rho=0.39 (p=2e-5), and H4 (no floor) leaves
  T-structure (p=0.02): each added global term removed the specific held-out
  structure that motivated it.
* H7 vs H6 on LOCO is -0.007 +/- 0.012 (jackknife): statistically
  indistinguishable.  H7 is named winner on the residual-structure criterion
  (H6 fails the no-S-structure requirement) and on AIC/BIC; H6 is the
  fallback if one insists on one fewer parameter.
* Collapse figure: `collapse_H7.png` -- after stripping scale, (1-mu),
  M_eff, corrected-tilt and S terms, all 112 points from every convention and
  scale fall on the single curve log2(T^-alpha + f).  The visible low cluster
  at T=5 is G5B.

## Frozen criteria of INCLUSION.md: does anything earn "law" status?

**No.**  Applied to the full fit of the winner: fixed-effect heterogeneity
chi^2 = 108,309 on 97 dof (p ~ 0, needs p >= 0.05) and marginal 95%-interval
coverage 10.4% of noise-eligible points (needs >= 90%).  Every other model is
worse or comparable on these criteria (best coverage anywhere: 14.2%, H4).
The seed-level intervals in this ledger have median half-widths of a few
percent of eta*, while the best cross-campaign model still mispredicts a
typical held-out point by ~28% (0.36 bits).  The residual variance is
campaign-level (G5B alone contributes a ~1.3-bit offset no admissible model
can absorb), so under the frozen rules the honest label for H7 is
**best-so-far descriptive**, not LAW.  H7 reduces the baseline's held-out
error by 3.8x (1.373 -> 0.363 bits) and removes all detectable T/mu/S/
convention structure, but a claim of "law" would require prospective,
independently-tuned campaigns to land inside seed-level intervals, which
these banked data already rule out.

Own-design provenance (H4-H7 were added after inspecting round-1 residuals,
as permitted by the task): the pre-registered eight are B0, P1, Q1, Q2, X1,
X2, R1, R2; H1-H3 are first-round hybrids; H4/H5 respond to the convex-in-
log-T mu0 residual, H6 to the T=160 upturn (seen in two independent
campaigns), H7 to the S trend unmasked by H6.  All refinements were selected
on held-out (LOCO) error, never on full-fit error.

---

# Round 2: theory-lane candidates (C1 / C1+C3) scored in the same harness

Added per coordinator instruction from `mech/law-v2/theory/CANDIDATES.md`
(commit 0dde35e).  Same data, same equal-point objective, same LOCO protocol,
same fallbacks, same seed.  New kinematic factor: `T/C_conv(T,mu)` with the
accumulated per-step coefficient from
`lean-mechanism/LeanMechanism/FiniteHorizonOuter.lean`:
`C_raw = T/(1-mu) - mu^2(1-mu^T)/(1-mu)^2`,
`C_hb = T/(1-mu) - mu(1-mu^T)/(1-mu)^2`, `C_corr = T/(1-mu)`, `C_mu0 = T`.
The C3 interference factor `phi = exp(-beta sqrt(H) eta0[scale] base(T)
(C-T))` needs `eta0` inside the exponent, so those two models carry the three
scale intercepts explicitly in theta (same parameter count, same objective);
beta is ONE global parameter refit inside every LOCO fold -- the theory
lane's beta=0.006 was never used as input.

## Updated league (new entries interleaved; full 21-model table in `league.csv`)

| model | form | k (global+stratum) | LOCO RMSE | full RMSE | AIC | BIC | Delta vs B0 (jk SE) | 95% cov | het chi2 (dof) |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| **C1sat-H7kin** (round-2 league winner) | H7 with (1-mu)M_eff -> theorem T/C; rho0,d dropped | 7 (3+4) | **0.331** | 0.294 | **-257.9** | **-236.1** | -1.043 (0.091) | 9.4% | 162k (99) |
| C1sat-H7kin+align | + H7's rho0 d^T (M-1) kept on top of T/C | 9 (5+4) | 0.331 | 0.295 | -253.8 | -226.6 | -1.042 (0.091) | 9.4% | 162k (97) |
| C1sat+C3 | eta0[s] (T/C)(T^-a+f) exp(-beta sqrt(H) eta0 base (C-T)) | 6 (3+3) | 0.346 | 0.303 | -253.2 | -234.1 | -1.028 (0.091) | 10.4% | 166k (100) |
| C1sat-pure | eta0[s] (T/C)(T^-a+f) | 5 (2+3) | 0.357 | 0.319 | -243.9 | -227.5 | -1.016 (0.084) | 10.4% | 233k (101) |
| H7-rho-floor-S (round-1 winner) | fitted alignment decay | 9 (5+4) | 0.363 | 0.317 | -237.2 | -210.0 | -1.010 (0.098) | 10.4% | 108k (97) |
| C1q+C3 | eta0[s] (T/C) q^T exp(-beta ...) | 5 (2+3) | 0.619 | 0.577 | -111.3 | -95.0 | -0.754 (0.063) | 0.9% | 490k (101) |
| C1q-pure | eta0[s] (T/C) q^T (theory keystone form) | 4 (1+3) | 0.692 | 0.649 | -87.0 | -73.4 | -0.681 (0.060) | 2.8% | 661k (102) |

Fitted round-2 parameters: C1sat-H7kin alpha=1.195, f=0.143, sigma=0.070,
beta_c=0.093; C1sat+C3 alpha=1.225, f=0.161, **beta=0.0021**; C1q+C3
q=0.9951, **beta=0.0042** (theory lane's independent estimate: 0.006 -- same
order, fold-refit, no leakage); C1q-pure q=0.9947.

## Per-campaign LOCO RMSE deltas vs H7 (bits; negative = C-variant better)

| campaign | H7 | C1sat-H7kin | delta | C1sat+C3 | delta |
|---|---:|---:|---:|---:|---:|
| DISAMBIG | 0.126 | 0.210 | +0.084 | 0.281 | +0.155 |
| G12 | 0.518 | 0.248 | -0.270 | 0.253 | -0.265 |
| G4C (1.7B) | 0.408 | 0.567 | +0.159 | 0.634 | +0.226 |
| G5B | 1.317 | 1.220 | -0.097 | 1.237 | -0.080 |
| G6 | 0.234 | 0.217 | -0.017 | 0.256 | +0.022 |
| G8 | 0.396 | 0.271 | -0.125 | 0.265 | -0.131 |
| G9A (1.7B) | 0.122 | 0.386 | +0.264 | 0.487 | +0.365 |
| G9B (7B) | 0.640 | 0.494 | -0.146 | 0.495 | -0.145 |
| TP-pilot | 0.414 | 0.132 | -0.282 | 0.102 | -0.312 |
| TP-v1 | 0.146 | 0.295 | +0.149 | 0.259 | +0.113 |
| TP-v2 | 0.153 | 0.246 | +0.093 | 0.258 | +0.105 |
| TP-v3 | 0.192 | 0.247 | +0.055 | 0.228 | +0.036 |

Pooled: C1sat-H7kin beats H7 by 0.032 bits = **1.1x** the jackknife SE
(0.029) -- a real-looking but NOT individually significant pooled gain under
the round-1 margin standard; the wins concentrate in the momentum-heavy
campaigns (G12 heavy-ball, G8, TP-pilot, G9B) and the losses in the two 1.7B
campaigns (G4C, G9A) -- exactly where the theory lane's scale-lever
signature lives.

## Structure tests, best C-variant (C1sat-H7kin; plots
`collapse_winner.png`, `residuals_holdout.png`)

Held-out residuals: Spearman vs T rho=0.021 (p=0.82) and vs S rho=0.020
(p=0.83) -- clean; **but vs mu rho=-0.467 (p=2.1e-7) and Kruskal-Wallis
across conventions p=1.7e-5** -- NOT clean.  The pattern: mu0 rows sit
systematically high (+0.16..+0.41 bits at T>=5) while momentum rows sit low
(heavy-ball -0.25, raw at mu=0.95 -0.21), i.e. under the parameter-free T/C
transform the momentum arms are still overpredicted by a roughly
convention-blind ~0.3-bit deficit that the theory lane calls phi.  H7 showed
no such structure (mu p=0.20, convention p=0.25) because its two extra
fitted parameters (rho0, d) bend the transform to absorb exactly this.  The
C3 term is the theory lane's proposed phi closure, and with the saturating
clock it only reduces LOCO by 0.012 +/- 0.012 bits (n.s.) and does not
remove the mu/convention structure (C1sat+C3 medians: mu0 +0.22 vs momentum
-0.25..0).  On the q^T clock C3 is clearly real (-0.073 +/- 0.023, >3x SE,
beta=0.0042), but the saturating floor f and the self-quenching phi are
partially degenerate on banked data -- both flatten the large-T tail.

## Frozen coverage/heterogeneity bars

No variant approaches them.  Best coverage in the entire 21-model zoo:
14.2% (H4); round-2 winner 9.4%; heterogeneity chi2 stays O(10^5) on ~100
dof (p ~ 0) everywhere, versus the frozen bars of >=90% coverage and
p>=0.05.  The verdict category remains **best-so-far descriptive** for
every model, rounds 1 and 2 alike.

## Honest read on C1's status: **confirmed as the kinematic factor, with a
measured residual transform still missing -- "mixed, leaning confirmed"**

1. **The theorem-backed kinematics subsumes my fitted alignment-decay
   term.**  Refit with H7's alignment factor kept on top of T/C, the
   optimizer drives it to irrelevance (rho0=1.26 at d=0.05, the lower
   bound: the factor is <=0.008 bits at T=2 and 0 beyond) and held-out
   error is unchanged (+0.0002 +/- 0.0008 bits).  H7's phenomenological
   4.07 x 0.657^T alignment decay was a curve-fit shadow of T/C: the
   theorem explains where round 1's best fitted structure came from, with
   zero parameters.  This is the cleanest positive result of the round.
2. C1 kinematics + refit saturating clock beats every round-1 model on
   AIC/BIC by >20 points with two fewer parameters, and posts the best
   pooled LOCO (0.331); the pooled gain over H7 itself is 1.1x SE (not
   individually significant).
3. **But T/C is not yet the complete convention transform**: it leaves the
   significant, roughly convention-blind mu/momentum deficit described
   above, which H7's fitted parameters were absorbing.  The theory lane's
   own phi ledger predicted exactly such a shared residual; its C3 closure
   is the right order of magnitude (fold-refit beta 0.002-0.004 vs the
   theory's 0.006) but is statistically inseparable from the floor f on
   banked data, and neither removes the structure.
4. **G5B outlier:** unchanged in kind under every C-variant -- still the
   worst campaign by ~3x (1.18-1.24 bits vs 1.32 under H7; a slight
   improvement, direction consistent with the kinematics helping its raw
   arm, but the ~1.2-bit campaign-level offset survives all admissible
   models).  C1/C3 do not explain G5B.
5. The 1.7B regressions (G4C +0.16, G9A +0.26 bits vs H7) are the banked
   shadow of the theory lane's scale-lever discriminator; the banked 1.7B
   data are too thin to settle it, which makes the proposed corrected-1.7B
   (T=20, H=512) curve the right next probe.

Round-2 verdict on the league: **C1sat-H7kin is the new best-so-far
descriptive law** (best LOCO, best AIC/BIC, fewest assumptions per bit of
error removed, theorem-backed kinematics), with the explicit caveat that it
fails the round-1 "no remaining structure" clause that H7 passes -- the
remaining structure is now localized, measured (~0.3 bits, mu/convention),
and is the object the C3/C4 discrimination experiments should target.
