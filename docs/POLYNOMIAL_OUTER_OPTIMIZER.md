# The Robust Polynomial Outer Optimizer

Status: 2026-07-12. Research direction + first diagnostic + first implementation.
Companion to docs/LEAN_THEOREMS.md (the impossibility this escapes),
docs/NEXT_OPTIMIZER_PLAN.md (the memoryless-spatial track this refines), and
scripts/diagnose_outer_dynamics.py (the measurement that chose the family).

## Reframing: the outer loop is numerical integration of a stiff field

The DiLoCo outer loop is a discrete integrator of the merged-pseudo-gradient
field `g(theta)`:

    theta_{t+1} = theta_t - lr * Optimizer(g_t, state).

Fixed-momentum Nesterov is then a **fixed-step, fixed-damping, infinite-memory
explicit integrator**: its first-moment buffer carries old velocity into
directions that have since become steep. When the pseudo-gradient geometry
shifts round-to-round (short horizon, changing curvature), that retained
velocity is exactly the transverse displacement Lean T1/T2 prove is
curvature-gated poison, and Lean T3 proves no geometry-blind scalar cap can
undo. Three controller generations (rho-adaptive v1/v2, capped-nesterov and its
gc/r/curv/wsub variants) confirmed this empirically: none beat memoryless
SGD-0.28.

The escape is **not another momentum cap**. It is to change the integrator:
either **memoryless spectral acceleration** (no buffer at all, only a scalar
step schedule shaped by curvature) or **short-memory trust-region / Krylov
updates** that first *identify the local dynamics* and then take one informed
step. Both classes **acquire local curvature/dynamics information** — the very
input Lean T3's geometry-blind scalar controller is assumed to lack — so the T3
impossibility no longer directly applies (see "Relation to Lean T3" below).

## The six candidate families

1. **Chebyshev-SGD (memoryless spectral acceleration).** No momentum buffer.
   Apply a short cyclical LR schedule `small -> large -> mid -> small` around the
   tuned base 0.28 whose K-step product is the shifted-Chebyshev residual
   polynomial for a symmetric operator with eigenvalues in `[lambda_min,
   lambda_max]`. State = one scalar cycle-phase counter per fragment. Restart the
   cycle on a large geometry change; degenerate to plain SGD when the spectrum is
   isotropic. Lowest cost, lowest risk. **Implemented (see below).**
2. **Trust-Krylov (short-memory local-dynamics ID).** Fit the local dynamics
   operator in the span of the recent 3-5 outer deltas, solve a tiny `3x3`-`5x5`
   system (never `d x d`), take a trust-region-gated step. Anderson / GMRES /
   multisecant lineage. Directional memory is bounded (a handful of vectors) and
   is *used to identify*, not to bias, the step.
3. **Robust-control FIR synthesis.** A short FIR controller
   `a0 g_t + a1 g_{t-1} + a2 g_{t-2}` synthesized over a measured curvature/noise
   uncertainty set, with a hard gain bound and a constraint that
   noise-amplification `<=` SGD. Safety by construction.
4. **Anisotropic thermostat (per-mode friction).** Strong friction on sharp
   modes, weak on flat, with the sharp subspace estimated from Hessian-vector
   products / worker disagreement / secants. Suppresses exactly the T2 sharp-mode
   contamination.
5. **Implicit / midpoint / extragradient / proximal.** For **rotational /
   non-conservative** fields, where an explicit step over-/under-shoots because
   motion in one direction induces orthogonal change. Warranted only when the
   antisymmetric part of the dynamics operator is large.
6. **Low-rank Gauss-Newton.** Adapter-subspace GN with a per-worker Fisher
   sketch. Highest cost and highest ceiling.

## Flagship: the Robust Polynomial Outer Optimizer

**Chebyshev spectral acceleration (1) + Krylov local-dynamics ID (2) +
robust-control safety (3), with an SGD-0.28 fallback.** It escapes T3 by acting
on the *measured curvature/dynamics* of the field rather than a gain/cosine
scalar of `(g_t, buffer)`.

### Relation to Lean T3 (stated honestly)

Lean T3 proves that a controller which sees only `(g, b, mu)` and outputs a
scalar step-scale on the fixed direction `d = g + mu b` cannot dominate tuned SGD
across **all** quadratics. The polynomial optimizer sidesteps the hypothesis:
it does not carry a foreign direction `b` (families 1/3/4 have no first-moment
buffer; family 2's memory is used only to *estimate* `A`, not to add velocity),
and its step-scale is a function of **curvature** (the operator `A` / its
spectrum), which T3's controller is defined not to see. So T3 does not apply to
it. **But the honest limit remains:** if you demand a win over the *entire*
quadratic family with a *fixed* schedule, the adversary picks a curvature the
schedule is miscalibrated for, and the safety fallback forces you back to SGD —
T3's shadow. The achievable, defensible target is therefore: **beat SGD within
the measured finite range of curvature, rotation, and noise this product regime
actually exhibits, with strict never-worse-than-SGD safety.** That range is what
the diagnostic below measures.

## The diagnostic that chose the family (scripts/diagnose_outer_dynamics.py)

Over a sliding window of recent secant pairs `(Delta_theta_t, Delta_g_t)` we fit
a low-dimensional local dynamics operator `A` (`Delta_g ~= A Delta_theta`,
regressed in the Krylov subspace of the recent deltas via a thin `L x L` QR —
never `d x d`) and read three scale-invariant properties of `A`:

- **Q1 SPECTRAL WIDTH** — condition number of the symmetric part `S = (A+A^T)/2`
  (flat-vs-steep spread). Wide -> Chebyshev / Krylov / Gauss-Newton.
- **Q2 ROTATION** — `||W||_F / ||S||_F` of the antisymmetric part `W =
  (A-A^T)/2`, plus complex-eigenvalue content of `A`. Large -> implicit /
  extragradient / midpoint / proximal.
- **Q3 SHARP-MODE CONCENTRATION** — participation ratio of `|eig(S)|` (effective
  number of active modes). Low -> anisotropic thermostat / low-rank suppression.

The pseudo-gradient is production-exact: for mu=0 (open-loop) captures
`g_t = (theta_t - theta_{t+1}) / lr` is recovered from consecutive anchor
checkpoints alone (the syncer's verified RDA replay), needing no candidate
payloads; for mu>0 captures `g_t` is the per-tensor RDA merge of the four learner
candidates. Data are the retained EXP2 captures (no GPU): rank2 mu0 H16/H64/H256
(local anchor cache), rank16 mu0 H16/H64 and inner-lr lo/hi H64 mu09
(exp2-35-generality S3).

### Results (medians over sliding windows; full stats in
experiment-results/EXP2/outer-dynamics-diagnostic/summary.json)

| regime | axis | Q1 cond | Q2 rot | Q3 PR | family |
|---|---|---:|---:|---:|---|
| rank2-H16  | horizon  | 20.2 | 0.59 | 3.3 | Chebyshev/Krylov |
| rank2-H64  | horizon  | 15.5 | 0.55 | 3.5 | Chebyshev/Krylov |
| rank2-H256 | horizon  |  7.7 | 0.53 | 2.1 | anisotropic/low-rank |
| rank16-H16 | rank     | 19.4 | 0.58 | 3.3 | Chebyshev/Krylov |
| rank16-H64 | rank     | 17.9 | 0.56 | 3.4 | Chebyshev/Krylov |
| innerlr-lo (5e-4, mu09) | inner-lr | 13.2 | 0.31 | 3.6 | Chebyshev/Krylov |
| innerlr-hi (2e-3, mu09) | inner-lr | 11.8 | 0.31 | 3.6 | Chebyshev/Krylov |

Reading:

- **Q1 is the dominant structure.** The symmetric spectrum is wide (in-subspace
  condition number ~15-20 at short/medium horizon; note this is a *lower bound*
  on the full-space conditioning, since it is measured in a 5-dimensional Krylov
  subspace). The field is stiff/anisotropic — exactly the regime where a
  Chebyshev polynomial schedule buys worst-case contraction over plain SGD. At
  the long horizon H256 the effective condition number falls (7.7) and the danger
  **concentrates** into ~2 modes (PR 2.1): the horizon axis trades spectral width
  for sharp-mode concentration, favoring a thermostat/low-rank suppressor there.
- **The LoRA-rank axis is geometry-invariant.** rank16 reproduces rank2 almost
  exactly (H16: cond 19.4 vs 20.2, rot 0.58 vs 0.59, PR 3.3 vs 3.3; H64: 17.9 vs
  15.5). The rank-2 rho-inflation objection does *not* extend to the local
  dynamics geometry — the same optimizer family is indicated at rank 16.
- **The inner-LR axis moves stiffness modestly and confirms Chebyshev.** At
  fixed H64/mu09, the low inner-LR (5e-4) arm is slightly stiffer (cond 13.2)
  than the high inner-LR (2e-3) arm (cond 11.8) — larger inner steps mildly flatten
  the effective spectrum — but both stay firmly in the moderate-width, spread,
  Chebyshev/Krylov regime. No inner-LR setting pushes the field toward isotropy
  (where cheb-sgd degenerates to SGD) or toward strong rotation.
- **Rotation is mild but nonzero.** In the mu0 (open-loop, anchor-difference `g`)
  regimes it is remarkably stable at ~0.53-0.59 — a persistent antisymmetric
  component of ~half the symmetric magnitude; in the mu09 (closed-loop, RDA-merge
  `g`) inner-LR arms it is lower (~0.31, near-conservative). The two are not
  apples-to-apples (different `g`-recovery method AND momentum), so the clean
  comparisons are within-method (rank2-vs-rank16 mu0; innerlr-lo-vs-hi mu09).
  Either way rotation is below the threshold that would mandate an
  implicit/extragradient integrator, but it is the single most important caveat
  for the Chebyshev bound, which is a symmetric-operator result (see below). Part
  of the mu0 ratio is fit noise / nonstationarity in a 5-sample window; its
  stability across regimes suggests a genuine floor.

**Recommended family: Chebyshev-SGD (family 1) as the low-cost first bet at
short/medium horizon, hardened toward the flagship (Chebyshev + Krylov local-ID +
robust-control safety), with anisotropic/low-rank suppression reserved for the
long-horizon concentrated regime (H256).** Implicit/extragradient (family 5) is
*not* indicated yet — rotation is present but sub-threshold; revisit if the
Chebyshev step is destabilized by the non-normal component.

## Implementation status: `cheb-sgd` (family 1, the #1 low-cost bet)

Implemented in `syncer/src/merge.rs` as the `cheb-sgd` outer optimizer.

- **Memoryless.** Each commit applies `theta -= lr * m_k * delta`, where `delta`
  is the current merged pseudo-gradient and `m_k` is a scalar cyclical
  multiplier. No first-moment buffer enters the direction; the optimizer buffer
  slot is overwritten each commit with `m_k * delta` (used only so the
  `materialize_applied_step` `lr * buf` branch stays bit-identical, and read once
  as a cosine probe for the restart). The only cross-round state is a **scalar
  cycle-phase counter per fragment**.
- **Cycle length K = 4**, multipliers `small -> large -> mid -> small` (Leja
  order).
- **Multipliers from the measured spectral width.** `cheb_sgd_multipliers(kappa)`
  builds the shifted-Chebyshev-node reciprocal schedule for eigenvalues in
  `[1, kappa]`, then anchors it by **arithmetic-mean** normalization (so the
  cycle-average multiplier is exactly 1 — the average LR stays the tuned 0.28)
  and hard-clamps to `[0.5, 2.0]` via a monotone `tau` solve that preserves the
  unit mean. `kappa = 1` gives `[1,1,1,1]` exactly (degenerate = plain SGD).
  `CHEB_SGD_KAPPA = 20` is calibrated to the diagnostic's short-horizon spectral
  width; at that value the bounded schedule is approximately
  `[0.50, 2.00, 1.00, 0.50]` (steps `[0.14, 0.56, 0.28, 0.14]` at lr 0.28).
- **Restart** resets the phase to 0 (smallest, safest step first) when the merged
  delta's cosine against the previous applied direction drops below 0 — a large
  geometry change that invalidates the `[1, kappa]` calibration.

Math (derivation cross-checked with codex gpt-5.6-sol): with `a=1, b=kappa,
c=(a+b)/2, d=(b-a)/2, q=cos(pi/8), r=cos(3pi/8)`, the ordered raw steps are
`s = [1/(c+dq), 1/(c-dq), 1/(c-dr), 1/(c+dr)]`, `m_j = tau*s_j` clamped so
`mean(m)=1`. Ideal (unclamped, symmetric-operator) 4-step contraction is
`rho_Cheb,4 = (kappa-1)^4 / (kappa^4 + 28 kappa^3 + 70 kappa^2 + 28 kappa + 1)`
vs `rho_SGD,4 = ((kappa-1)/(kappa+1))^4`, i.e. ~3.1x better residual contraction
per cycle at kappa=10, ~2x at kappa=30 — the theoretical ceiling, before noise,
nonstationarity, and the measured non-normality (rot ~0.55, which *weakens* the
symmetric-operator guarantee: bounds on the symmetric part alone do not imply the
same polynomial contraction for a substantially non-normal operator).

Tests (`cargo test`, green): multiplier degeneracy at kappa<=1, ordering +
bounds + unit-mean, deterministic 4-step cycle sequence, restart-resets-phase on
geometry change, and materialize bit-identity. Added to `OUTER_OPTIMIZERS` in
`scripts/compare_diloco.py` (`pytest -k outer` green).

## Next steps

1. GPU screen `cheb-sgd` vs SGD-0.28 across the sensitive settings (staged run
   spec; on-demand, prefix `exp2-44-cheb`), gated by the same product success
   bar as NEXT_OPTIMIZER_PLAN (paired win > 2x noise on >=2 core workloads, never
   worse than 1 noise floor, no per-H tuning).
2. If cheb-sgd wins at short/medium H but the non-normal component destabilizes
   it, add the family-2 Krylov local-ID to *measure* kappa online (replacing the
   fixed `CHEB_SGD_KAPPA`) and the family-3 robust gain bound; at H256, test the
   family-4 anisotropic/low-rank suppressor instead.
3. Only escalate to family 5 (implicit/extragradient) if Q2 rotation rises above
   threshold under the inner optimizers we must support (AdamW **and** Muon).
