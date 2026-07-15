# Cross-Disciplinary Lit Research: the momentum poison as a known failure mode

Codex sweep (math / physics / bio-control, subagents), 2026-07-13. Live web was
unavailable; citations are established-literature metadata. Verdict converges
with docs/BAKEOFF_RESULTS.md, docs/CTTN_DESIGN.md, docs/OTHER_OPTIMIZERS.md.

## The two uncannily-close analogues

### 1. MD timestep stability / velocity-Verlet (physics) — literal
The stale transverse momentum deposits energy into normal modes:
`Δθ_⊥ = −η μ² r` (r = P_⊥(b)), with positive quadratic energy error
`½ η² μ⁴ rᵀHr` on top of the proven `D_∥ᵀ H D_⊥` cross term. Velocity-Verlet's
stability limit `h√λ_i < 2` is **mode-curvature-dependent, not velocity-norm
dependent** — exactly why norm-based caps (wsub, Euclidean caps) fail. Our exact
Nesterov-mode boundary `ηλ_i < 2(1+μ_i)/(1+2μ_i)` tells the same story. The fix
handed down by Verlet / RESPA / critical damping: estimate `QΛQᵀ`, use SGD (or
smaller steps) in sharp modes, allow inertia only in the flat complement,
smoothly `z_i = r_i/(1+τλ_i)`. **That is literally CTTN.** The missing cheap
piece is acquiring `Q,Λ`.

### 2. Integral anti-windup (control theory) — the exact controller architecture
```
b_free = mu*b + g ;  d_free = g + mu*b_free
d_safe = Project_U(d_free)              # U = affine Hessian trust ellipsoid at SGD
b_new  = b_free + (d_safe - d_free)/mu  # back-calculation discharges the integrator
```
with `gᵀd_safe = ||g||²` and `(d_safe−g)ᵀH(d_safe−g) ≤ ρ gᵀHg`. With real `H`
this IS CTTN / a reference governor. With `H=I` it degenerates to the falsified
Euclidean cap. With no curvature sensor at all, the only universally safe
projection is `d_safe = g` — exactly SGD.

## What transfers / what doesn't
- Su–Boyd–Candès damped-oscillator ODE explains the ringing; Shi et al. high-
  resolution ODE exposes the missing `HẊ` anisotropic damping → motivates the
  SECANT candidates.
- Dynamic-regret (Besbes–Gur–Zeevi, Hall–Willett) supports discounting / finite
  windows / restart under drift but gives no uniform accelerated method when the
  variation is unknown.
- True RKC/ROCK/super-time-stepping needs `s` fresh stage evaluations of ~the
  same operator → several HVPs or fresh pseudo-gradient stages (NOT free). A
  cross-round Chebyshev LR cycle (our unrun cheb-sgd) is a worthwhile ZERO-COST
  control but is "not literally RKC once the field rotates/changes between rounds."
- IMEX / backward-Euler / exponential integrators give the right resolvent
  `(I+τH)⁻¹r` or `e^{−τH}r` but need an operator action/solve.

## Verdict — an OBSERVABILITY problem
From current `(g,b)`, norms, cosines, consensus, or scalar moments you CANNOT
distinguish two systems with identical observations but arbitrarily different
curvature along `r = P_⊥(b)`. A controller retaining unknown transverse memory
cannot be uniformly safe; it must either (a) ERASE r → become SGD, (b) observe a
delayed plant response (secant / inner-trajectory), or (c) query curvature (HVP).

A cheap **stabilizer** certainly exists (purge transverse memory = SGD). Evidence
for a cheap **uniform improver over tuned SGD** is weak. >1% overhead is not
mathematically unavoidable (secants/inner-trajectory reuse existing info), but
every established fix with a real stiff/anisotropic guarantee is implicit,
multi-stage, curvature-aware, or output-tested. CTTN's 8 HVPs ≈ 40% at H16, so
even one comparable HVP ≈ 5%, and a genuine extra pseudo-gradient stage is well
over the gate.

**Product verdict: ship memoryless SGD-0.28.** The only sub-1% experiments with a
credible chance of changing that, in order:
1. rank-one **secant IMEX-SGD** (implicit `(I+τ ĥ)⁻¹` on a rank-1 secant curvature),
2. normalized **K=4 inner-trajectory MOLLY** (use the inner path as the delayed
   plant sensor),
3. conservative rank-one **secant anti-windup**,
plus PR/SWA (tail averaging) as a separate low-risk final-model test.
The probability that another geometry-blind momentum controller clears the
multi-workload gate is very low.

## Convergence with the independent brainstorm (docs/OTHER_OPTIMIZERS.md)
Both arrive at: secant-based cheap methods (trust-Krylov / IMEX / anti-windup)
are the credible bets; Chebyshev is a zero-cost control (weak once the field
rotates — the secant GUARD is what matters); tail averaging is a cheap final-model
test; and CTTN is the theoretically-correct expensive oracle — triangulated
independently via physics (Verlet/critical damping) AND control (anti-windup with
real H). Honest base rate: SGD likely wins.
