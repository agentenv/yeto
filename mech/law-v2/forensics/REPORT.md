# law-v2 forensics: the empirical age transform r(T)

**Scope.** Pure characterization, no theory. For every stratum (convention x mu) with
matched momentum/mu=0 pairs in the 112-point ledger
(`mech/law-unification/tuned_optima.csv`), we extract

    r(T; mu, conv) = [ eta*(mu,T) / eta*(0,T) ] / (1 - mu)

so that the failed law's prediction is exactly `r(T) = M(T, mu)` (the code-true
multiplier), and test shape hypotheses on r(T) per stratum. eta0(scale) and q^T cancel
in the ratio by construction.

**Files.**
- `r_of_T_pairs.csv` — tidy per-pair ledger (66 pairs: 62 within-campaign, reproducing
  the published `paired_cancellation.csv` momentum points 1:1; + 4 clearly-flagged
  cross-campaign pairs that extend mu=0.5/0.8/0.95 coverage). Columns include r, CIs
  (conservative marginal endpoints), se_log_r, M_code, r/M, pair_type, noise_eligible.
- `r_of_T_cellmeans.csv` — inverse-variance cell means of r per (stratum, scale, T).
- `shape_fits.csv` — all shape fits, per stratum (chi2, GOF p, AICc, RMSE, params).
- `loco_stability.csv` — leave-one-campaign-out stability of the winning shapes.
- `s_dependence.csv`, `scale_residuals.csv` — S and scale forensics.
- `fig_r_of_T_raw_mu09.png`, `fig_r_of_T_all_strata.png`, `fig_s_dependence.png`.
- `forensics.py` + `run_log.txt` — fully reproducible (`python forensics.py`).

**Error model (honest-GOF convention).** Bootstrap SEs in this ledger are known to be
far too tight (upstream heterogeneity chi2 = 2.8e6/103 dof). All GOF numbers below use
sigma_eff^2 = se_boot^2 + s_rep^2, where s_rep is the *model-free* replicate floor
estimated from within-(stratum, T, scale) scatter across S-replicates/campaigns
(raw mu0.9: 0.054 nat = 0.077 bits; corrected mu0.9: 0.088 nat; strata without
replicates use the median floor 0.071 nat). `shape_fits.csv` also carries chi2 under
raw bootstrap SEs. Single-seed 7B points (point-mass CIs) never enter fits.

**Ledger facts used throughout** (verified in `run_log.txt`):
1. `H*T == S` for **all 112 points**. H is never an independent degree of freedom;
   any "S effect" at fixed T is equally an H effect.
2. Only G6 is a true T x S factorial. TP-v3/TP-pilot/G8 fix H=512 so S = 512*T
   (S confounded with T); TP-v1/v2 fix S=2560 so H = 2560/T. The fixed-S and fixed-H
   ladders land on the same r(T) curve (see LOCO), so T — not S or H — is the driver.
3. TP-v3 ran mu=0 in both raw and corrected lanes: the two lanes returned *identical*
   tuned rates (0.000 bits spread in all 4 cells) — they are one tuning outcome, and
   we pool them (IVW) into a single control per cell.

---

## 1. Headline: the raw-Nesterov age transform saturates to exactly 1

**Stratum nesterov_raw, mu=0.9** — the only stratum spanning T = 2..160
(24 within-campaign 135M pairs, 8 campaigns). Cell means:

| T | n | r (95% CI) | M_code | log2 r/M (bits) |
|---:|---:|---|---:|---:|
| 2 | 5 | 4.195 [3.988, 4.413] | 2.710 | +0.63 |
| 5 | 8 | 2.470 [2.377, 2.568] | 4.686 | -0.92 |
| 10 | 5 | 1.580 [1.507, 1.658] | 6.862 | -2.12 |
| 20 | 4 | 1.030 [0.976, 1.087] | 8.906 | -3.11 |
| 40 | 1 | 0.996 [0.851, 1.167] | 9.867 | -3.31 |
| 160 | 2 | 1.032 [0.957, 1.113] | 10.000 | -3.28 |

r(T) decays from ~4.2 at T=2 and then **pins at 1.000 from T~20 through T=160**
(T=40: 0.996; T=160: 1.032; both CIs cover 1). Operationally: for aged raw-Nesterov
runs the tuned optimum is just `eta*(mu,T) = (1-mu) * eta*(0,T)` — the momentum
multiplier washes out completely.

Shape-hypothesis scoreboard (n=24, sigma_eff):

| rank | model | k | chi2/dof | GOF p | params |
|---:|---|---:|---:|---|---|
| 1 | **sat_exp_to_one**: log r = ln(A) e^(-T/tau) | 2 | **1.20** | **0.23** | A=6.71, tau=6.74 |
| 2 | saturating_exp (free asymptote) | 3 | 1.22 | 0.22 | r_inf=0.977, A=6.81, tau=6.93 |
| 3 | power_law c*T^-a | 2 | 17.9 | 1e-69 | c=4.29, a=0.369 |
| 4 | exp_decay c*rho^T | 2 | 61.0 | 2e-270 | c=2.18, rho=0.9941 |
| 5 | constant | 1 | 80.0 | ~0 | c=1.91 |
| 6 | multiplier x decay c*M(T)*rho^T | 2 | 209 | ~0 | c=0.436, rho=0.9888 |
| 7 | scaled multiplier c*M(T) | 1 | 279 | ~0 | c=0.339 |
| 8 | law r=M(T) | 0 | 643 | ~0 | — |

The **only** surviving 2-parameter form is a saturating exponential *in log r* with the
asymptote at exactly 1; freeing the asymptote adds nothing (r_inf = 0.977, consistent
with 1). Everything with a multiplier term is catastrophically rejected, and so are
pure power law / pure exponential once the T=40/160 plateau is in view (a power law
fits fine on any T<=20 sub-window — which is exactly why short-window strata below
cannot discriminate).

**LOCO stability** (drop each campaign, refit sat_exp_to_one): A in [6.64, 6.88],
tau in [6.55, 6.93], chi2/dof in [0.87, 1.36]. No single campaign carries the shape.
Note A = 6.7 is materially *less* than the full multiplier 1/(1-mu) = 10, i.e. even
extrapolated to T->0 the pairs never exhibit the code-true multiplier.

**Two-regime / crossover verdict:** there is **no multiplier regime at small T**.
The two-regime model ("M(T) up to Tc, decay after") collapses to Tc ~ 1 (degenerate
with exp_decay, chi2/dof 64) and even at T=2 the observed r=4.19 is off M(2)=2.71 by
+0.63 bits. The only real "crossover" is r(T) reaching 1: the fitted transform is 50%
washed out at T = tau*ln2 ~ 4.7 rounds and is statistically indistinguishable from 1
for T >~ 20.

## 2. Other raw strata (T<=20 windows, weaker but consistent)

| stratum | cells r(T) | best shapes (GOF p) | notes |
|---|---|---|---|
| raw mu=0.5 (T=5*,10,160) | 1.13*, 1.08, 1.00 | constant~1 (p=0.49) | already saturated at 1 by T>=10; consistent with sat-to-one with small tau. *T=5 is a cross-campaign pair. |
| raw mu=0.8 (T=2,5,20; G8) | 2.30, 1.60, 0.81 | power law a=0.45 (p=0.99); exp rho=0.949 (p=0.011); sat-to-one A=4.1,tau=3.9 (p=0.002) | **undershoots 1 at T=20** (CI 0.71-0.93 excludes 1) — single G8 point; the one tension with a universal saturate-to-1 story. |
| raw mu=0.95 (T=2,5,20; G8) | 8.01, 4.82, 1.04 | all 2-param marginally rejected (p 0.007-0.015); free sat-exp saturates (r_inf 0.55, tau 12.5, 0 dof) | reaches ~1 at T=20; needs T>20 to discriminate. |
| heavy_ball mu=0.9 (T=2,5,20; G12) | 6.28, 2.97, 1.15 | sat-to-one A=12.7, tau=6.0 (p=0.43); power law a=0.73 (p=0.41); exp rejected (p 3e-6) | same tau~6 as raw mu0.9; amplitude 12.7 vs 1/(1-mu)=10. |

Fitted tau (sat-to-one) vs mu for raw-like conventions: mu=0.9: 6.7; heavy-ball 0.9:
6.0; mu=0.95: 7.1 (rejected fit; free-asymptote tau 12.5); mu=0.8: 3.9 (rejected fit).
For reference only: -1/ln(mu) = 4.5 / 9.5 / 19.5 at mu = 0.8 / 0.9 / 0.95. The T<=20
strata cannot pin tau(mu); only mu=0.9 is well measured.

## 3. Corrected-Nesterov strata decay *away* from 1 (slow leak, shape open)

Corrected arms have M=1, so the failed law predicts r(T)=1. Instead r starts near ~0.94
at T=2 and keeps drifting down — the mirror image of raw (which decays *toward* 1 from
above). Saturate-to-one is rejected outright (mu=0.9: p=1e-8); constant is rejected
(p=8e-8). On T<=20, exponential decay, power law, and free-asymptote saturating exp are
all consistent and cannot be separated:

| stratum | cells r(T) | exp_decay rho (p) | power_law a (p) | free sat-exp |
|---|---|---|---|---|
| corr mu=0.8 (T=2,5,20) | 0.944, 0.863, 0.597 | 0.9752 (0.88) | 0.204 (0.30) | saturated, 0 dof |
| corr mu=0.9 (T=2..20, n=18) | 0.942, 0.794, 0.707, 0.606 | 0.9780 (0.30) | 0.189 (0.59) | r_inf=0.58, tau=8.2 (p=0.50) |
| corr mu=0.95 (T=2,5,20) | 0.954, 0.820, 0.411 | 0.9546 (0.91) | 0.376 (0.04) | saturated, 0 dof |

Notable regularities (characterization only): (i) corrected decay is nearly identical
for mu=0.8 and mu=0.9 at every age, and steepens at mu=0.95; (ii) the corrected
per-round leak rho ~ 0.975-0.978 at mu<=0.9 matches the WRITEUP clue (stratum
q_corrected 0.928 < q_mu0 0.995 in WLS; in ratio space the leak is milder because the
paired estimand removes eta0/q sharing artifacts); (iii) the free-asymptote fit at
mu=0.9 has tau ~ 8, close to the raw tau ~ 7 — suggestive of one age scale, but T<=20
cannot confirm it. **Discriminating exp vs power vs saturating for corrected arms
requires corrected pairs at T >= 40** — the single most valuable new measurement.

## 4. S-dependence: confirmed second-order, but real, negative, and age-coupled

WRITEUP's claim "S is not first-order" is **confirmed in ratio space and sharpened**.
In the only unconfounded design (G6 T x S factorial, S in {2560, 5120, 10240}), with T
fixed effects:

| stratum | bits per doubling of S | 95% CI | z | p |
|---|---:|---|---:|---:|
| raw mu=0.9 | -0.089 | [-0.144, -0.034] | -3.2 | 0.0014 |
| corrected mu=0.9 | -0.106 | [-0.195, -0.017] | -2.3 | 0.020 |

So the paired ratio r is not exactly S-free: momentum optima tune slightly *lower*
relative to their mu=0 controls as S grows. But the effect is ~0.1 bits/doubling
(~0.2 bits over the tested 4x range) against 2-3+ bits of T-structure — second order.
Two forensic sharpenings:
- **The S slope grows with age** (raw: -0.02, -0.06, -0.10, -0.17 bits/doubling at
  T = 2, 5, 10, 20; corrected: +0.02, -0.10, -0.14, -0.21). At T=2 there is no S effect
  at all; treat "S effect" as an age-interaction, not a main effect.
- Because H = S/T identically everywhere, this is indistinguishable from an H effect
  at fixed T. No design in the bank separates S from H.
(The WRITEUP's +0.0976 +/- 0.38 bits was the *absolute* residual pooled across arms —
our paired estimand is sharper and differs in sign; both agree it is not first-order.)

## 5. Scale-dependence: r(T) is scale-portable at the +/-0.5 bit level

The ratio construction cancels eta0(scale), so any scale dependence of r is a genuine
law-v2 term. Non-135M pairs vs the 135M sat-to-one curve (raw mu=0.9):

| point | T | residual (bits) |
|---|---:|---:|
| G4C 1.7B T5 | 5 | -0.51 |
| G4C 1.7B T20 | 20 | +0.22 |
| G9B 7B T5 (single seed, point-mass CI) | 5 | +0.10 |

Mixed signs, |res| <= 0.51 bits, n=3: **no coherent scale trend**; the age transform
transfers across 135M -> 1.7B -> 7B far better than any multiplier model transfers
across age. (G9A 1.7B T10 has no mu=0 control anywhere at its cell — unpairable.)

## 6. Ranked candidate functional forms for the law-v2 age transform

1. **Saturating exponential in log-rate ratio, asymptote exactly 1** (raw-family):
   `eta*(mu,T) = eta*(0,T) * (1-mu) * exp( ln A(mu,conv) * exp(-T/tau) )`,
   tau ~ 6-7 rounds at mu=0.9 (raw and heavy-ball), A < 1/(1-mu).
   Evidence: only surviving shape on the full T=2..160 window; GOF p=0.23; LOCO-stable;
   scale-portable. Weakness: raw mu=0.8 T=20 undershoot (one point).
2. **Slow multiplicative leak for corrected arms**: r(T) = rho^T with
   rho ~ 0.975-0.978 (mu<=0.9), 0.955 (mu=0.95) — *or* equally a T^-0.2 power law or a
   saturating form with r_inf ~ 0.5-0.6; undetermined below T=40.
3. **Power law T^-a** — viable only inside T<=20 windows (it is the small-window
   shadow of form 1); decisively rejected (p ~ 1e-69) once T=40/160 exist.
4. **Any form containing the code-true multiplier M(T)** — rejected everywhere, at all
   ages, in every stratum (best case chi2/dof ~ 12). There is no age at which the
   banked optima follow M(T); the multiplier never "turns on".

Secondary terms for a v2 law, in order of size: age-coupled S(=H*T) term
(~ -0.09 to -0.11 bits/doubling at mu=0.9, growing with T, zero at T=2); scale term
(< ~0.5 bits, sign-incoherent, n=3 — likely ignorable at first order).

*Reproduce with `python forensics.py` (needs pandas, numpy, matplotlib; chi2 tails
computed in-script).*
