# law-v2 VERIFIER: adversarial audit of the C1 kinematic-law claim

**Claim under attack** (coordinator formulation): *"eta\* = eta0[scale] ·
saturating-clock(T) · T/C_conv(T,mu) is the correct kinematic law
(theorem-backed by `finiteHorizon_optimum_alignment`); it subsumes the best
fitted alignment structure with zero parameters; a real ~0.3-bit
mu/convention residual phi remains, consistent with theory candidate C3
(beta ~ 0.004-0.006)."*

**Verifier artifacts** (all reproducible from repo root, same harness, same
seed, same LOCO protocol as the zoo):

- `attack_experiments.py` -> `attack_results.json` (wrong-kinematics
  injection, forensics-shape competitor, leave-both-TP-out floor probe,
  non-135M residual audit)
- `signed_align_probe.py` -> `signed_align_results.json` (the one-sidedness
  refutation of the subsumption test)
- `delta_significance.py` / `delta_significance2.py` -> jackknife SEs
- `nested_selection_cv.py` -> `nested_selection_cv.json` (selection-honest
  LOCO)
- `fig_r_overlay.png` (forensics r(T) vs C1's implied r(T), raw mu=0.9)

Zoo league numbers were independently re-derived in this audit and match
RESULTS.md (C1sat-H7kin LOCO 0.331, H7 0.363, +align 0.331).

---

## Verdict summary

| # | Sub-claim | Verdict |
|---|---|---|
| 1 | The Lean theorem states what is claimed; the C_conv formulas in `fit_zoo.py` match the Lean definitions | **CONFIRMED-WITH-CAVEATS** (raw/hb exact to 2e-13; `C_corr` is *not in the Lean file*; the theorem is a frozen-gradient displacement-matching identity, not a derivation of tuning behavior) |
| 2 | "T/C subsumes H7's fitted alignment structure" | **REFUTED as evidence** (the subsumption test was structurally one-sided; a signed version of the same family revives at kappa = -0.09 and beats C1sat-H7kin by 2.1x jackknife SE) |
| 3 | "eta0 · sat-clock · T/C is the correct kinematic law" | **REFUTED as "the" law; CONFIRMED-WITH-CAVEATS as a competitive theorem-motivated anchor** (beaten held-out by the forensics-shape kinematics, which also passes every residual-structure test C1 fails; free exponent prefers (T/C)^1.10; fails T=160 extrapolation when both T=160 campaigns are held out) |
| 4 | "a real ~0.3-bit mu/convention residual phi remains" | **CONFIRMED** that phi is real and roughly convention-shared, **with caveats**: it is 0 to -1.2 bits, strongly mu- and T-structured (dip at T ~ 20-24), not a flat ~0.3-bit offset |
| 5 | "phi is consistent with C3 (beta ~ 0.004-0.006)" | **REFUTED at the stated strength** (beta is clock-dependent by 3x and n.s. on the winning clock; C3 misses the deepest banked cells by ~0.5 bits; the measured dip location matches the C2/C4 1/(1-mu) clock, not C3's drift clock; the banked 1.7B G9A point contradicts C3's eta0 scale lever) |

---

## 1. Theorem check (symbol-by-symbol)

`lean-mechanism/LeanMechanism/FiniteHorizonOuter.lean` (imported by the root
module; `.olean` build artifact present, dated Jul 25; no `sorry`).

**What the theorem actually says.** `finiteHorizon_optimum_alignment`
(L489-511): for `rule in {heavyBall, textbookNesterov}`, `T > 0`,
`mu >= 0`, on the 1-D quadratic `(1/2) a (theta - eta c g)^2` with the
gradient *frozen* at `g` for all `T` outer updates, the loss-minimizing
rates at momentum `mu` and at `mu = 0` satisfy
`etaMu * C(T,mu) = etaZero * C(T,0)` with `C = effectiveCoeff` and
`C(T,0) = T` (`effectiveCoeff_zero_momentum`). So `etaMu = etaZero * T/C`:
the form the zoo uses. `finiteHorizon_any_minimizers_align` extends this to
any global minimizers. Verified: this is exactly the claimed statement.

**Formula check.** Brute-force `effectiveCoeff` per the Lean *definitions*
(sum of `terminalMultiplier` at ages 1..T) agrees with `fit_zoo.py`'s
`accumulated_C` closed forms to max |err| = 2.3e-13 over
T in {1..160} x mu in {0, .5, .8, .9, .95}:

- `C_raw = T/(1-mu) - mu^2(1-mu^T)/(1-mu)^2` = `nesterovCoeff_closed_form` (L187-189). Match.
- `C_hb = T/(1-mu) - mu(1-mu^T)/(1-mu)^2` = `heavyBallCoeff_closed_form` (L171-173). Match.
- `C_mu0 = T` = `effectiveCoeff_zero_momentum` (L203). Match.
- `M` in `fit_zoo.finite_age_M` = `terminalMultiplier` closed forms (raw
  exponent T+1, hb exponent T). Match.

**Caveats (real, not fatal):**

1. **`C_corr = T/(1-mu)` is NOT in the Lean file.** The file has no
   corrected rule at all (only heavyBall / textbookNesterov). The form is a
   trivially correct one-liner *if* the production correction divides the
   age-t direction by `(1-mu^{t+2})` (verified numerically: per-step
   coefficient is then exactly `1/(1-mu)` at every age), but it is an
   unformalized extension, and it is also *exactly the failed v1 factor*
   for the corrected arm (`T/C_corr = 1-mu`, same as `(1-mu)*M` with
   `M = 1`). The entire corrected-arm age drift (r: 0.94 -> 0.61 over
   T = 2..20, forensics section 3) is carried by the *fitted* `beta_c`
   tilt and by phi in every C-variant. "Zero parameters in the transform"
   is true only for raw/hb.
2. **The theorem is kinematics + a modeling bridge, not a law derivation.**
   At its optimum the frozen-gradient loss is exactly 0 for *any* nonzero
   coefficient; the theorem holds equally for any candidate `C`. Its
   empirical content is (a) the code-true recursion fact that total
   displacement is `C * eta * g` (`accumulatedDisplacement_closed_form`),
   and (b) the *assumption* that real tuned optima are total-displacement
   matchers. (b) is not theorem-backed and is precisely what phi measures
   the failure of.
3. Since `T/C = 1` for mu0 and `1-mu` for corrected (both conventions
   identical to v1), the theorem's testable content lives entirely in the
   raw (42 pts) and heavy-ball (3 pts) arms.

## 2. Circularity: the subsumption test had no power, and its conclusion flips under a signed probe

The zoo's headline mechanism claim ("H7's fitted 4.07 x 0.657^T alignment
decay was a curve-fit shadow of T/C; refit with the alignment kept on top,
the optimizer drives it to irrelevance") rests on `C1sat-H7kin+align`,
whose added factor is `1 + rho0 d^T (M-1)` with bounds `rho0 in [0, 64]`
(fit_zoo.py L651-653). Because `M >= 1`, this factor is **>= 1
identically**: it can only *raise* momentum-arm predictions.

But C1sat-H7kin's own held-out residuals show the momentum arms are
**over-predicted** at mid-T (mu0 rows +0.16..+0.41 bits, momentum rows
-0.21..-0.25; Spearman-vs-mu rho = -0.467, p = 2.1e-7). The correction
needed on top of T/C is *below* 1. The subsumption test therefore could
not have revived the alignment term **no matter whether T/C is right**:

- Injection control (`attack_experiments.py`): with deliberately wrong
  kinematics `(T/C)^0.7` — or the kinematics deleted entirely
  (`gamma = 0`) — the same pipeline *also* drives the alignment term to
  zero (footprint 0.000 bits). The test "passes" wrong kinematics exactly
  as it passed the right one. (Only over-suppression, `gamma = 1.3`,
  partially revives it, footprint 0.37 bits — the one direction the
  one-sided family can express.)
- Signed probe (`signed_align_probe.py`): allow the same decay family with
  free sign, `delta_bits = kappa * d^T * (M-1)` on top of T/C. It revives
  decisively: `kappa = -0.091`, `d = 0.984`, **LOCO 0.279** vs C1sat-H7kin
  0.331 (**-0.051 bits, 2.1x jackknife SE**) and vs H7 0.363 (-0.083,
  2.4x), and it *removes* the mu/convention structure (Spearman-mu p =
  0.65, convention KW p = 0.18) that C1sat-H7kin fails at p ~ 1e-5..1e-7.
  An equivalent hump parameterization (`Vhump-phi`, peak at T ~ 24) does
  the same (LOCO 0.279, all structure tests clean).

**Verdict: the "subsumption with zero parameters" result is an artifact of
a one-sided test family.** What survives: (i) *dropping* H7's rho0/d after
inserting T/C does not hurt pooled LOCO (0.331 vs 0.363) — a genuine
parsimony win; (ii) T/C is a *better basis* than (1-mu)M: the same 2-param
signed decay works better on top of T/C (0.279) than it did inside H7
(0.363). But T/C does not subsume the fitted structure — it replaces
roughly the small-T half of it and leaves a ~0.5-bit signed mid-T
correction that H7's parameters were carrying.

## 3. Leakage audit

**Protocol internals are clean.** Read of `fit_zoo.py` confirms: all theta
refit inside every fold; profiled intercepts from training rows only;
`make_start` uses the training fold only; the two fallbacks are declared
and model-independent; no per-campaign parameters anywhere.

**But structure selection sits outside the folds, twice:**

1. **Form-design leakage (acknowledged in RESULTS.md, unquantified
   there).** H4-H7 — including the saturating clock `(T^-alpha + f)` that
   round 2 grafts C1 onto — were designed after inspecting all-campaign
   LOCO residuals. The pre-registered-only league tops out at R1 = 0.607;
   everything below that number is post-hoc form search on the same 12
   campaigns.
2. **Winner-pick leakage, quantified here and largely EXONERATED**
   (`nested_selection_cv.py` -> `nested_selection_cv.json`): selecting
   the league winner by inner LOCO on the 11 training campaigns of each
   outer fold (all 21 zoo models), then scoring the pick on the held-out
   campaign, gives a selection-honest pooled RMSE of **0.334 bits vs the
   in-sample-selected 0.331** — model-pick optimism is ~0.003 bits,
   negligible. The inner folds pick a C1sat-family model in **12/12**
   folds (C1sat-H7kin or +align 11x, C1sat+C3 once, H7 never), so within
   the given menu the C-family win is fold-stable, not a lucky in-sample
   pick. The leakage that remains is the *menu itself*: the clock form
   (item 1) and the exclusion of the forensics-shape kinematics, which
   beats every menu member when added (section 4).
3. **The floor does not survive its provenance test.** The T=160 upturn
   that motivated the floor f exists in exactly two sibling campaigns
   (TP-v1/TP-v2, same fixed-S=2560 design); ordinary LOCO always keeps one
   in training. Holding *both* out: H7 still predicts their 15 points to
   RMSE 0.148 (T=160 cells +0.13 bits mean), while **C1sat-H7kin misses
   T=160 by +0.69 bits mean** (pair RMSE 0.425; C1sat-pure +0.58, C1sat+C3
   +0.47). Under T/C the refit clock flattens (alpha 1.20 -> 1.01) to
   compensate the mid-T phi deficit, and large-T extrapolation collapses.
   The round-2 winner's good T=160 numbers are conditional on a T=160
   sibling campaign in training; the round-1 winner's are not. For a
   claimed *kinematic law*, failing the only long-horizon extrapolation
   test in the bank while the phenomenological rival passes it is a
   serious strike.
4. Round-2 specifics are fair-by-symmetry (same folds, same clock family
   for H7 and C1 variants), and the zoo already flags that the 0.032-bit
   C1-vs-H7 win is 1.1x SE, below its own declared 3x standard.

## 4. Forensics consistency: two celebrated shapes that disagree by 0.5 bits in the middle

Overlay (`fig_r_overlay.png`), raw mu=0.9, r(T) = paired momentum/mu0
tuned-rate ratio / (1-mu); forensics sat-exp uses its published A = 6.71,
tau = 6.74; C1 is parameter-free:

| T | r obs (95% CI) | forensics sat-exp | C1: (T/C)/(1-mu) | obs vs C1 (bits) |
|---:|---|---:|---:|---:|
| 2 | 4.195 [3.99, 4.41] | 4.116 | 4.338 | -0.05 |
| 5 | 2.470 [2.38, 2.57] | 2.476 | 2.971 | **-0.27** |
| 10 | 1.580 [1.51, 1.66] | 1.540 | 2.117 | **-0.42** |
| 20 | 1.030 [0.98, 1.09] | 1.103 | 1.552 | **-0.59** |
| 40 | 0.996 [0.85, 1.17] | 1.005 | 1.249 | **-0.33** |
| 160 | 1.032 [0.96, 1.11] | 1.000 | 1.053 | -0.03 |

- The two shapes agree at the endpoints (<= 0.08 bits at T=2 and T=160)
  and disagree by up to **0.49 bits at T = 10-20** — 5-8x the stratum
  replicate floor (0.077 bits). They are *not* interchangeable
  descriptions; they are reconciled only by attributing the entire mid-T
  gap to phi. Equivalently: **forensics' tau ~ 6.7 exponential washout and
  C1's ~ 1 + 8.1/T power-law tail are different curves, and the banked
  data sit on the forensics curve** (that is what it was fit to; GOF
  p = 0.23, while C1-alone is rejected at 3.5-7.7 sigma per mid-T cell
  under sigma_eff).
- The celebrated T=160 asymptote "three-decimal hit" does not
  discriminate: at T=160 C1 actually predicts ratio 0.1053 (not its
  T->inf limit 0.100); observed 0.1027/0.1036 sits between C1 and the
  sat-exp prediction (0.1000), within ~0.04 bits of both.
- **Head-to-head in the zoo's own harness** (`VFsat-fixedA`: forensics
  kinematics with amplitude pinned at the steady multiplier
  `ln A = ln(1/(1-mu))`, one shared refit tau = 5.7, applied to raw/hb):
  **LOCO 0.305 vs C1sat-H7kin 0.331** (-0.025 bits, 1.6x jackknife SE;
  vs H7 -0.058, 2.3x), full RMSE 0.267 vs 0.294, **and it passes every
  residual-structure test** (T p=0.98, mu p=0.76, S p=0.36, convention
  p=0.32) that C1sat-H7kin fails (mu p=2.1e-7, convention p=1.7e-5). By
  round 1's own winner criterion — best LOCO *plus* no remaining
  structure, the exact clause used to prefer H7 over H6 — the round-2
  winner should have been the forensics-shape kinematics, not C1.
- A free kinematic exponent (`(T/C)^gamma`) refits to gamma = 1.098 and
  also beats plain T/C held-out (0.313, 1.6x SE): given one degree of
  freedom, the data walk away from the theorem's gamma = 1.
- Symmetric-leakage caveat, stated plainly: the sat-exp *form* was itself
  fit by the forensics lane on these same campaigns (tau is refit inside
  every fold here, so its status is exactly that of H6's floor — a
  data-suggested form scored honestly). The verifier's point is not that
  VFsat is the true law; it is that the banked data cannot distinguish
  the two kinematics and, on the zoo's own two criteria, currently
  *prefer the rival*. Note also the split at the ends: at T=2 the pinned
  amplitude A = 1/(1-mu) overshoots (5.06 vs obs 4.20, +0.27 bits) where
  C1's 4.34 is nearly exact — C1 wins the endpoints, the sat-exp shape
  wins the dense mid-T interior.
- What C1 does win, genuinely and parameter-free: the T=2 amplitudes
  (raw 4.34 vs obs 4.20; hb 6.90 vs obs 6.28 — forensics' shared-A form
  cannot produce the hb/raw difference without an extra parameter), the
  cross-convention ratio `C_raw/C_hb` = 1.59 vs measured 1.51 at (T=2,
  mu=0.9), the mu=0.5 near-null, and the G12 heavy-ball fold (LOCO 0.25
  vs H7's 0.52: the parameter-free `C_hb` transfers to a convention never
  seen in training better than H7's raw-fitted alignment decay does).
  These are real, and they are why C1 deserves anchor status at small T
  even though it loses the shape contest.

## 5. The 1.7B losses and G5B

- Non-135M momentum data: **4 points** (G4C raw T=5/T=20, G9A raw+corr
  T=10, G9B raw T=5). Any claim of scale-generality of the kinematics
  rests on these.
- Point-level LOCO residuals: the G4C losses under C-variants are
  *not* kinematic — the worst move is the mu0 T=5 point (+0.75 -> +1.07
  bits), where T/C = 1 by definition (intercept/clock effect). But
  **G9A raw T=10 is kinematic territory and moves -0.05 (H7) ->
  -0.52 bits (C1sat-H7kin)** — the same ~-0.4..-0.5-bit mid-T deficit the
  135M raw arm shows at T=10. G9B 7B raw T=5 improves slightly under C1
  (-0.69 -> -0.57).
- Consequence for C3 (see 6): the deficit at 1.7B is as deep as at 135M
  in the one clean banked cell, while C3 predicts it should shrink by
  ~6x (eta0 lever). Direction favors a scale-free phi (C4-flavored) in
  that cell; G4C's paired T=20 point (-0.28 vs -0.70) points the other
  way. 4 points cannot settle it — but they mean **"C1's wins are a
  135M-plus-G12 result; its scale story is untested"** is the accurate
  summary, and the mid-T deficit definitely does not vanish at 1.7B.
- G5B: ~-1.2..-1.3-bit campaign-level offset under every admissible model
  (slightly smaller under C-variants, 1.22 vs 1.32). Orthogonal to the C1
  question; no C-variant explains it; correctly excluded from the margin
  argument.

## 6. The phi residual and the C3 attribution

phi is real and roughly convention-shared at matched cells (raw -0.59 /
hb -0.52 / corr -0.72 bits at mu=0.9, T=20). But:

- It is **not "~0.3 bits"**: it spans 0 (mu=0.5, and T=160) to **-1.19
  bits** (raw mu=0.95, T=20; corr mu=0.95 T=20 is -1.28), with a
  T-dip at ~20-24 and strong mu-growth. "~0.3 bits" is the median over a
  ledger dominated by mu=0.9 mid-T cells.
- **beta is not stable**: theory-lane pair fit 0.0060; zoo q^T clock
  0.0042; zoo saturating clock (the league winner) **0.0021 and
  statistically inseparable from zero gain** (0.012 +/- 0.012). The claimed
  0.004-0.006 band excludes the value the winning model actually fits.
  The floor f and phi are degenerate on banked data, as the zoo itself
  notes — so banked data cannot certify the C3 form.
- C3's own 62-pair fit (reproduced here: beta = 0.00595, rms 0.186 bits)
  leaves *systematic* structure: every raw mu=0.9 row -0.13..-0.28 bits,
  the deepest cell (raw mu=0.95 T=20) missed by **-0.49 bits**, and the
  T=160 recovery over-suppressed (+0.16/+0.18). Its success is
  concentrated in the corrected G6 block.
- **Dip location cuts against C3**: the verifier's hump fit puts the phi
  dip at T ~ 24 for mu=0.9 (visible in the raw table too: max deficit at
  T=20, recovery by T=40 already at H=64). C3 pins the dip at
  T* ~ -1/ln q (~ 80 at the stratum q = 0.9876; ~ 200 at the round-2
  q = 0.9951); C2/C4 pin it at ~ (1.2-2)/(1-mu) = 12-20 at mu=0.9. The
  measured location matches the 1/(1-mu) clock, not the drift clock.
- CANDIDATES.md's own narrative overstates its evidence: it says phi
  "vanishes at (T=160,H=16) **and (T=40,H=64)**", but its own script
  output gives phi(T=40,H=64) = 0.798 — a 0.33-bit deficit, 2.8x its se.
  Only the T=160 recovery is real.

So: a shared residual exists (as the claim says), but its attribution to
C3 at beta ~ 0.004-0.006 is not supported by the banked data, and two of
the three shape diagnostics (dip location, 1.7B depth) currently point
away from C3.

---

## Final verdicts

1. **Theorem-backing: CONFIRMED-WITH-CAVEATS.** The Lean theorem says what
   the theory lane claims and the zoo's C formulas match it symbol-for-
   symbol (raw/hb to 2e-13). Caveats: `C_corr` is an unformalized (if
   trivially checkable) extension; the theorem is a frozen-gradient
   displacement-matching identity whose bridge to real tuned optima is a
   modeling assumption, and its testable content lives only in the raw and
   heavy-ball arms.
2. **"Subsumes the fitted alignment structure with zero parameters":
   REFUTED.** The test was one-sided by construction (align factor >= 1;
   needed correction < 1); it passes wrong kinematics identically; the
   signed version revives at kappa = -0.09 and beats the round-2 winner by
   2.1x SE while clearing the mu/convention structure. T/C is a better
   *basis* than (1-mu)M, but ~0.5 bits of signed, decaying,
   momentum-specific structure that H7's parameters carried is NOT in T/C.
3. **"The correct kinematic law": REFUTED as stated / downgrade to
   "competitive theorem-motivated anchor, small-T and cross-convention".**
   Four independent strikes: (i) residual mu/convention structure at
   p ~ 1e-5..1e-7 where the criterion demands none; (ii) beaten held-out
   by the forensics saturating-exponential kinematics (0.305 vs 0.331,
   1.6x SE) which passes all structure tests — by round 1's own winner
   rule the C1 variant is not the best model in the bank; (iii) a free
   exponent walks away from the theorem value (gamma = 1.10, LOCO 0.313);
   (iv) T=160 extrapolation fails (+0.69 bits) once both T=160 campaigns
   are held out, while H7 passes (+0.13). What survives: best AIC/BIC per
   parameter, zero-parameter T=2 amplitudes, the hb/raw ratio, mu=0.5
   null, and the G12 heavy-ball fold — real predictive content no rival
   matches parameter-free.
4. **"Real ~0.3-bit mu/convention residual, consistent with C3":
   CONFIRMED-WITH-CAVEATS for existence, REFUTED at the stated strength
   for the C3 attribution.** The residual is 0..-1.2 bits with a
   T ~ 20-24 dip; beta collapses to 0.002 and n.s. on the winning clock;
   dip location and the one clean 1.7B cell currently favor a
   1/(1-mu)-clocked, scale-free phi (C4-flavored) over C3's
   drift-clocked, eta0-scaled form. C3 remains alive only because the
   banked designs confound H with T and the floor with phi.

**Strongest surviving attack** (the one the C1 camp must answer): *by the
zoo's own two-part winner criterion — held-out error plus no residual
structure — the theorem kinematics loses to the forensics saturating-
exponential shape in the same harness (0.305 vs 0.331, structure-clean vs
structure-failing), and it loses the only available long-horizon
extrapolation test (+0.69 vs +0.13 bits at T=160 with both T=160 campaigns
held out). A kinematic factor that must borrow a ~0.5-bit signed empirical
correction precisely where the data are densest, and that corrupts the
clock's extrapolation when denied a sibling campaign, is not yet "the
correct kinematic law" — it is the theory-preferred member of a family the
banked data cannot separate, currently second-best on the data's own
terms.*

## Cheapest decisive new-data experiment

**One tuning cell: 135M, T = 40, mu = 0.9, H = 512 (S = 20480), three arms
tuned independently: mu0, nesterov_raw, heavy_ball.** (Same protocol as
G6/G8; ~one G8-sized curve per arm.)

Why this cell beats every alternative per GPU-hour:

- **Kinematic tail, at last unconfounded.** The banked T=40 evidence is
  ONE pair at H=64. At T=40, mu=0.9 the predictions for
  r = [eta(raw)/eta(mu0)]/(1-mu) split wide open: forensics sat-exp
  **1.00**; C1-pure **1.25**; C1+C3 at H=512 (theory beta = 0.006)
  **~0.61**; C4-flavored persistent phi (deficit frozen at its T=20,
  H=512 level) **~0.78**. With the demonstrated cell-mean precision
  (se ~ 0.04-0.10 bits), one cell separates all four at >= 3 sigma
  pairwise (closest pair: C4 vs C1+C3, ~0.35 bits apart).
- **H-lever on phi at fixed T,mu:** banked (T=40, H=64) has phi = 0.80;
  C3's sqrt(H) law predicts phi(H=512) = 0.53 x further drop, C2's floor
  forbids anything below 0.86 x, C4 predicts an H-grown but T-stable
  deficit. Same cell, no extra runs.
- **Heavy-ball arm doubles as C1's sharpest parameter-free signature**
  (CANDIDATES.md's own proposed severe test): C1 predicts
  eta(hb)/eta(raw) = C_raw/C_hb = **1.026** (near-equality) at T=40; the
  v1 M-law predicts a large gap; and any hb-vs-raw phi difference at
  matched (T,mu,H) falsifies the "convention-blind phi" premise shared by
  C1+C3 and C1+C4.
- It simultaneously extends the hb stratum past T=20 (currently 3 points,
  T <= 20), where the forensics tau ~ 6 vs C1 1/T-tail disagreement is
  0.3+ bits.

Secondary (if a second cell is affordable): corrected 1.7B (T=20, H=512),
the theory lane's scale-lever cell — it discriminates C3 vs C4 (phi 0.91
vs 0.54) but does not test the C1 tail; the T=40 triple does both.

Note: the zero-training Hessian discriminator
(`mech/law-v2/discriminator/protocol.md`, frozen 2026-07-28) attacks
C3-vs-C4 *given* C1 on banked checkpoints. It is free and should run
first, but no checkpoint probe can test the C1 kinematic tail itself —
only a T >= 40 momentum/mu0 pair at high H can, which is why the T=40
triple above remains the cheapest decisive *new-data* experiment for C1
proper.
