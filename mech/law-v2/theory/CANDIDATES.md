# law-v2 theory: candidate outer tuning laws (THEORY lane)

Status: candidate derivations + banked-data re-analysis only. No training was
launched. Every number below is recomputed from the frozen
`mech/law-unification/paired_cancellation.csv` ledger by the four scripts in
this directory (`c_normalization_check.py`, `matched_filter_check.py`,
`ar1_fit.py`, `rotation_fit.py`, `interference_fit.py`); run them with plain
`python3`, no third-party packages required.

Estimand used throughout: the matched-pair residual
`phi := observed_to_law_ratio / [candidate kinematic factor / ((1-mu) M)]`,
i.e. what is left of the same-(T,S) momentum/mu0 tuned-rate ratio after
dividing out a candidate age transform. `eta0(scale)` and `q^T` cancel exactly
in these pairs, so phi isolates the convention/age transform the v1 audit
flagged as the break (WRITEUP.md "Residual mechanism target").

---

## C1 (keystone kinematic law): accumulated-displacement alignment
**Replace the terminal multiplier M(T,mu) by the per-step-averaged accumulated
coefficient C(T,mu,conv)/T.**

```
eta*(scale, T, mu, conv) = eta0(scale) * q^T * T / C_conv(T, mu)

C_raw(T,mu)  = T/(1-mu) - mu^2 (1-mu^T)/(1-mu)^2      (nesterovCoeff_closed_form)
C_hb(T,mu)   = T/(1-mu) - mu   (1-mu^T)/(1-mu)^2      (heavyBallCoeff_closed_form)
C_corr(T,mu) = T/(1-mu)                                (correction: constant per-step coeff)
C_mu0(T)     = T
```

Parameter count: identical to v1 (one `eta0` per scale + one shared `q`); the
transform has **zero** free parameters. mu0 and corrected strata are
unchanged (`T/C = 1` and `1-mu` respectively); only raw Nesterov and
heavy-ball move.

Derivation: this is not a new model — it is the theorem already proved in
`lean-mechanism/LeanMechanism/FiniteHorizonOuter.lean`. The frozen-gradient
optimum equalizes the **total displacement over the fragment**
(`finiteHorizon_optimum_alignment`, `finiteHorizon_any_minimizers_align`):
`eta*_mu * C = eta*_0 * T`. The v1 law instead used the final-round
multiplier `M = terminalMultiplier`, which agrees with `C/T` only at `T=1`.
The Lean file itself warns that `C` grows like `T/(1-mu)` while `M`
saturates at `1/(1-mu)` (`effectiveCoeff_mul_one_sub_tendsto_atTop` vs
`terminalMultiplier_tendsto_steady`) — that divergence **is** the observed
"ratio decays while the multiplier saturates" anomaly (established fact (b)).

Evidence from banked data (`c_normalization_check.py`):
- The paired-cancellation spread in the mu=0.9 cells collapses from **5.03
  bits (M-law) to 0.89 bits (C-law)**.
- Conventions merge: at matched (mu=0.9, T) the raw / corrected / heavy-ball
  median residuals agree within <=0.23 bits at every age (T=20: 0.680 /
  0.594 / 0.698) versus a 2.44-bit raw-vs-corrected divergence under the
  M-law (-3.077 vs -0.634 bits). The "convention-specific q" (0.990 / 0.928 /
  0.831) is under C1 an artifact of stratum-specific (T,H) designs absorbing
  the shared residual phi (below) into per-stratum slopes.
- Parameter-free asymptote: C1 predicts `eta*_raw/eta*_mu0 -> (1-mu)` as
  T→infinity, i.e. observed_to_law -> 1-mu (the M-law predicts -> 1). Banked
  T=160 points: mu=0.9 give **0.1027, 0.1036** (predict 0.100); mu=0.5 gives
  **0.5018** (predict 0.500). This is a three-decimal, zero-parameter hit on
  the points the v1 law misses worst.
- Parameter-free cross-convention test: heavy-ball/raw tuned-ratio at same
  (T=2, mu=0.9) predicted `C_raw/C_hb = 4.61/2.90 = 1.59`; measured 1.51.

**Sharp falsifiable signature:** at any fresh (T,mu), the matched
momentum/mu0 tuned ratio equals `T/C` up to the *shared, convention-blind*
residual phi(T,mu,H). Any pair of conventions measured in the same cell must
give the same phi (the M-law predicts their residuals diverge without bound
in T). Cheapest severe test: one heavy-ball vs raw pair at (T=40, mu=0.9) —
C1 predicts their tuned-rate ratio `C_raw/C_hb = 1.026` (near-equality),
while the v1 M-law predicts `M_hb/M_raw`-shaped inequality plus the fitted
q-gap (0.99 vs 0.83 per step, i.e. a >8x disagreement at T=40).

After C1, the remaining shared residual phi has this measured structure
(fuel for C2-C4): phi≈1 at mu=0.5 everywhere; at fixed (T,mu) the deficit
scales ~sqrt(H) (G6 corrected T=20: -0.45/-0.63/-0.88 bits at H=128/256/512,
within 2% of a sqrt law); at fixed H=512 it deepens ~linearly in T through
T=20; it vanishes at (T=160, H=16) and (T=40, H=64).

---

## C2 (angle 3, noise-vs-signal EMA): buffer noise-energy closure — DERIVED AND REJECTED
**The tuned rate balances aligned displacement against accumulated
pseudo-gradient fluctuation energy.**

With p_t = g + n_t, forward-tail coefficients w_j (noise injected at outer
step j, carried down the remaining horizon by the buffer), and any stationary
fluctuation kernel K (white: K=sigma^2 I; AR(1): K_jk = sigma^2 rho^|j-k|):

```
phi(T,mu; s,rho) = (1 + s*Q0/T^2) / (1 + s*Q/C^2),   s = sigma^2/g^2
Q  = sum_jk w_j w_k rho^|j-k|,   Q0 = same with w=1 (mu0 arm)
w_j(raw) = 1 + mu(1-mu^{T-j+1})/(1-mu);  w_j(hb) = (1-mu^{T-j+1})/(1-mu);
w_j(corr) = sum_{t>=j} [delta_tj + mu^{t-j+1}]/(1-mu^{t+1})   (the bias
correction UP-scales early-step noise — this is why the corrected arm dips
despite a constant deterministic coefficient).
```

Parameters: one s per (scale,H) stratum, one rho shared (measurable
independently from the lag-1 tapes, 0.55-0.71 band). Crossover prediction:
phi dips at `T* ≈ 1.2-2/(1-mu)` (moves with mu as ~1/(1-mu)) and recovers to
1 at large T; at s→infinity phi is bounded below by the **Cauchy-Schwarz
floor** `C^2/(T W)`, W = sum w_j^2.

**Verdict on banked data (`matched_filter_check.py`, `ar1_fit.py`): rejected
as the phi closure.** The floor at (raw, mu=0.9, T=20) is 0.905 and at
(corrected, mu=0.9, T=20) is 0.766, versus observed 0.596/0.531; at
(mu=0.95, T=20) floor 0.86 vs observed 0.44. Positive rho only *shallows*
the dip (correlated fluctuation becomes common-mode and cancels in the
ratio), so the joint (s,rho) fit drives rho→0 and still leaves -0.6 to -1.0
bit misses. This retro-explains the earlier scalar buffer-norm rejection
(chi2=103, implied rho 0.994): **no second-moment closure over pseudo-
gradient fluctuations can reach the observed depth**; the earlier fit needed
an absurd rho because it was the wrong functional family, not the wrong
parameter. Signature kept for completeness: if any fresh cell ever lands
*below* `C^2/(T W)`, every variance-family law (including re-tunings of the
v1 buffer-norm closure) is dead there too — the banked G6/G8 T=20 cells
already do this.

---

## C3 (new; improves angle 1): first-order inner-state interference (self-quenching drag)
**Each outer update displaces parameters under an inner optimizer whose
adaptive state was fit to the pre-update basin; the re-adaptation drag is
first order in the displacement and diffusive in inner steps. Only the
excess per-step displacement relative to the mu0 control survives in the
matched ratio.**

```
log phi(T,mu,conv,H,scale) = -beta * sqrt(H) * eta0(scale) * q^T * (C_conv(T,mu) - T)
```

One new **global** parameter beta (shared across conventions, mu, H, T, and
scale; all structure enters through known quantities: sqrt(H), the control
arm's tuned rate eta0*q^T from the primary fit, and the Lean coefficient C).
Fit on all 62 banked pairs: beta = 0.0060, rms residual **0.186 bits**
(vs 0.278 C1-only, 0.224 best variance closure, 0.271 rotation closure); the
corrected mu=0.9 stratum — the one that breaks v1 hardest — fits to within
+-0.07 bits at every T>=5 and every H (`interference_fit.py`).

Why each factor: `(C - T)` = total *excess* displacement of the momentum arm
over its mu0 control (zero at mu=0 — explains the exact mu=0.5 null at
matched (T,H); ~ T*mu/(1-mu) for corrected, ramp-suppressed for raw/hb —
explains why corrected dips deeper than raw at T>=10, observed). `sqrt(H)` =
diffusive mismatch between the inner Adam moments and the post-jump basin —
matches the measured sqrt(H) column law. `eta0 q^T` = the drag is
proportional to the *actual step size*, so the deficit **self-quenches as
the tuned rate drifts down** — this is the only candidate that produces the
observed (T=160,H=16) and (T=40,H=64) recoveries without a hand-inserted
bump function.

**Sharp falsifiable signatures (distinguishing C3 from C4):**
1. *Scale lever (cheapest decisive test):* the deficit is proportional to
   eta0(scale). At (corrected, mu=0.9, T=20, H=512), C3 predicts 1.7B phi ≈
   0.91 (-0.14 bits) because eta0(1.7B)/eta0(135M) ≈ 1/6, while C4 predicts
   the deficit is scale-free (≈ -0.9 bits, same as 135M). Banked hint: G4C
   1.7B T=20 shows -0.28 bits vs 135M -0.70 in the same cell (direction
   right; the G4C T=5 point goes the other way and is the audit's known
   worst-family outlier — treat as open).
2. *Inner-optimizer ablation:* swap the inner optimizer to plain SGD
   (no adaptive state) in one G6 cell — C3 predicts the deficit collapses
   toward the C2 floor (>=0.9); C4 predicts it survives.
3. *Dip location:* C3's dip in T sits at T* ≈ argmax q^T (C-T) ≈ -1/ln q
   (~80 at the 135M q; moves with the *drift rate* q, hence with scale),
   NOT at 1-2/(1-mu). C4 and C2 pin the dip to ~1/(1-mu) (moves with mu).

---

## C4 (angle 2): curvature ratchet / edge-of-stability equilibration
**Tuned rate is pinned to an arm-specific stability edge while sharpness
grows with the work done: eta*_arm(T) = theta_conv(mu)/lambda_arm(T), with
progressive sharpening driven by per-step displacement:
d lambda / d step ∝ (step norm)^2, integrated over H inner steps per round.**

Since C1 equalizes *total* displacement, the momentum arm concentrates it in
larger late steps: its sharpening rate exceeds the control's by
`(T/C)^2 * E/T = 1 + CV^2(m_t)` (E = sum m_t^2) plus the buffer-noise step
variance, and its stability prefactor is theta_hb = 2(1+mu) vs theta_0 = 2.
Closed form (2 parameters: lambda-growth rate gamma per unit squared-step,
shared; theta ratios fixed by the conventions):

```
phi(T,mu,conv,H) = [1 + gamma*H*eta0^2*T] / [1 + gamma*H*eta0^2*T*(1+CV^2(m))*kappa_noise] * theta_conv/theta_0
```

Predicts deficit growing with H (more sharpening per outer step), growing
with mu at fixed T (bigger CV of the ramp, bigger buffer variance), and —
critically — **persisting or deepening at large T at fixed H** (sharpening
ratchets; nothing anneals it), with the dip/knee pinned near T ≈ 2/(1-mu).

**Sharp falsifiable signature (free — no training):** measure lambda_max on
the already-banked G6 (T=20, mu=0.9) momentum and mu0 checkpoints. C4
*requires* `lambda_mom/lambda_mu0 ≈ theta_ratio / phi ≈ 1.9/0.55`; C1+C3
predict lambda parity across arms (deficit lives in the inner-state, not the
Hessian). A Hessian power-iteration probe on existing checkpoints separates
C4 from C3 with zero new runs. Second signature: at (raw, mu=0.9, T=160,
H=512), C4 predicts phi <= 0.55 (ratchet never releases) — the same cell
where C2 predicts >= 0.98; C3 predicts an intermediate 0.46 *only if* the
stratum q is the global 0.9876, and >0.8 under the raw-stratum q, so the
triple (this cell, the scale lever, the Hessian probe) is jointly decisive
across all three closures.

Angle-1 note (pseudo-gradient decorrelation as *rotation*): the endpoint
form was derived and tested (`rotation_fit.py`: complex contraction
mu·e^{i·theta}, theta = theta0·H^p) and is near-null on banked data (best
rms 0.271 bits ≈ no improvement over C1 alone; the matched-filter endpoint
optimum is second-order insensitive to small buffer-staleness rotation). A
pathwise version survives only as the microscopic origin story for C3's
drag term; the tape test — regress log phi on tape-measured lag-1 rho within
strata — remains the way to kill or keep it, since C3 requires the drag to
track sqrt(H) even where rho is flat.

---

## Recommended discrimination matrix (for the EXPERIMENT lane)

| Probe | cost | C1+C2 | C1+C3 | C1+C4 |
|---|---|---|---|---|
| Hessian lambda_max on banked G6 T=20 checkpoints | zero (no training) | parity | parity | ratio ≈ 1/phi ≈ 1.8-3.4 |
| corrected 1.7B, T=20, H=512 | one curve | phi>=0.77 | phi≈0.91 | phi≈0.54 |
| raw+corrected mu=0.9, T=160, H=512 | expensive | phi>=0.98 | 0.46-0.85 (q-stratum-sensitive) | phi<=0.55 |
| inner-SGD swap in one G6 cell | one curve | unchanged | deficit collapses | deficit survives |
| heavy-ball vs raw at T=40, mu=0.9 | one pair | ratio C_raw/C_hb = 1.03 | same | same (C1 shared) |

C1 itself should be promoted to the law's kinematic factor regardless of
which of C3/C4 wins the residual: it is theorem-backed, zero-parameter, and
repairs facts (a), (b) and the T=160 asymptote simultaneously.
