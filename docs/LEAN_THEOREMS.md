# Machine-checked theorems: the transverse-momentum mechanism (2D quadratic)

Status: 2026-07-12. Lean 4 + Mathlib, sorry-free. Source:
`lean-mechanism/LeanMechanism/Basic.lean` (builds with `lake build`; Lean
toolchain `leanprover/lean4:v4.31.0`). Every theorem below compiles and
`#print axioms` reports only the three standard Mathlib axioms
(`propext`, `Classical.choice`, `Quot.sound`) — no `sorryAx`, no extra
assumptions smuggled in.

These compress the DiLoCo-poison empirical findings (docs/THEORY.md Result A/B,
the EXP2.37 matched-η_eff triads) onto a **minimal** anisotropic 2D quadratic —
deliberately *not* a full Local-SGD formalization. They isolate the one
mechanism the experiments kept surfacing: **outer directional (first-moment)
memory injects a transverse displacement whose harm is curvature-gated, and no
geometry-blind scalar controller can undo it.**

## Model (exact, no smoothness slack)

Loss `L(p) = ½ pᵀ H p`, `H` symmetric 2×2 with entries `a=H₁₁, b=H₂₂, c=H₁₂`.
Gradient `∇L(θ) = Hθ`. For a quadratic the one-step change of the update
`θ ↦ θ − u` is the exact identity (`dL` in the source):

    ΔL(u) = L(θ−u) − L(θ) = − uᵀ(Hθ) + ½ uᵀ H u.

Pseudo-gradient `g_t = ∇L(θ) = Hθ` (minimal model: even when the merged delta
equals the true gradient, the **buffer** `b_{t-1}` carries a foreign direction).
- SGD applies `u_sgd = η_s·g_t`.
- Nesterov applies `u_mom = η_m·d_t`, `d_t = (1+μ)g_t + μ²b_{t-1} = A·g_t + d⊥`
  with aligned gain `A = 1+μ+μ²c_t` and transverse part `d⊥ = μ²b⊥ ⟂ g_t`
  (docs/THEORY.md Prop A.1).
- "Matched aligned gain": choose `η_s = η_m·A` so the SGD step equals the
  parallel part of the momentum step, `u_sgd = Dpar = η_m A g_t`.

---

## T1 — Matched aligned gain does not imply matched descent  ✅ proved

`T1_descent_gap`, `T1_matched_sgd`.

Exact decomposition of the descent-gap: with equal aligned part `Dpar` and the
extra momentum part `Dperp ⟂ g` (`⟨Dperp,g⟩ = 0`),

    ΔL_mom − ΔL_sgd = Dparᵀ H Dperp + ½ Dperpᵀ H Dperp
                    = ½ ( Dperpᵀ H Dperp + 2 Dparᵀ H Dperp ).

This is the exact form the task asked for. `T1_matched_sgd` instantiates the full
Nesterov step with a transverse buffer (`c_t = 0`, `A = 1+μ`) and SGD at the
matched LR `η_s = η_m(1+μ)`, and proves *both* that the two steps share the
aligned part exactly (`u_mom = u_sgd + Dperp`, `Dperp = η_m μ² buf ⟂ g`) *and*
that their loss changes nonetheless differ by the transverse form above. So
matching the aligned projection provably does **not** equalize descent; the gap
is exactly the sharp-direction contamination the aligned match cannot see.

Product implication: an outer optimizer tuned to match SGD's *aligned* effective
step (the EXP2.25/2.37 η_eff matching) is not thereby matched in loss — the
residual is `½ Dperpᵀ H Dperp + Dparᵀ H Dperp`, a curvature quantity.

## T2 — Transverse momentum: benign under isotropy, harmful under anisotropy  ✅ proved

`T2_sgd_descends`, `T2_aniso_mom_ascends(_pos)`, `T2_iso_mom_descends(_neg)`,
`T2_threshold`.

Concrete verified instance (rationals; matched aligned step η_s = 57/200 ≈ 0.28,
the tuned-SGD LR; μ = 9/10; buffer purely in the sharp direction):

| quantity | value |
|---|---|
| `H` | `diag(1, ℓ_y)` |
| `θ = g` | `(1, 0)` (gradient in the flat direction) |
| buffer `b` | `(0, 1)` (transverse, sharp direction) |
| `Dpar = u_sgd` | `(57/200, 0) = η_m·A·g`, `η_m = 3/20`, `A = 19/10` |
| `Dperp` | `(0, 243/2000) = η_m·μ²·b` |

- Matched SGD descends by `19551/80000 ≈ 0.2444`, **independent of ℓ_y**
  (`T2_sgd_descends`).
- **Anisotropic ℓ_y = 100**: `ΔL_mom = 1974900/4000000 = +0.4937 > 0` — momentum
  turns a descent into an **ascent** (`T2_aniso_mom_ascends_pos`).
- **Isotropic ℓ_y = 1**: `ΔL_mom = −0.2370 < 0` — momentum still descends, the
  transverse harm is a negligible `0.0074` (`T2_iso_mom_descends_neg`).
- **Exact threshold** (`T2_threshold`): `ΔL_mom(ℓ_y) > 0 ⇔ ℓ_y > 3910200/118098
  ≈ 33.11`. Below it momentum is benign, above it harmful — the threshold-like
  behavior the matched-pairs experiment reported (negligible at mild curvature,
  severe at sharp).

Same buffer, same aligned step: only the curvature ratio flips momentum from
benign to catastrophic. This is the exact-arithmetic mechanism behind the
empirical "harm is threshold-like, severe under anisotropy" finding.

## T3 — No geometry-blind scalar cap dominates tuned SGD over all quadratics  ✅ proved

`bestReduction_is_scalar_optimum`, `T3_strict_gap`, `T3_no_uniform_bound`.

A controller applying a scalar step-scale `s(g,b)` to `d = g + μb` is stuck with
the **direction** `d`; only its length is free. `bestReduction_is_scalar_optimum`
proves that `bestReduction lx ly v g = ⟨v,g⟩²/(2 vᵀHv)` really is the supremum
over scalars `s` of the *actual* one-step loss reduction `−dL(s·v)`, attained at
`s⋆ = ⟨v,g⟩/vᵀHv` — so the impossibility below is about the true loss, not a
chosen formula.

- `T3_strict_gap`: for diagonal `H`, gradient `g=(gx,0)`, buffer `b=(0,by)`, the
  controller's best reduction along `d=(gx, μby)` is **strictly less** than tuned
  SGD's along `g`, whenever the transverse sharp mass is nonzero. The exact ratio
  is `ℓx gx² / (ℓx gx² + μ² ℓy by²) < 1`.
- `T3_no_uniform_bound` (the impossibility): for **every** target ratio
  `κ ∈ (0,1)` there is a sharp curvature `ℓy > 0` on which the controller's best
  reduction is `< κ ·` tuned SGD's. Since `g, b, μ` (hence any scalar `s`) are
  held fixed while `ℓy` varies, no fixed geometry-blind scalar controller stays
  within any constant factor of tuned SGD across the quadratic family.

This is the formal core of why three controller generations (rho-adaptive v1/v2,
capped-Nesterov, capped-Nesterov-gc/-r) failed: they scale `d_t` by a scalar
computed from gain/cosine (`g, b`) only, and that is provably insufficient.

## T4 — A curvature-aware transverse cap preserves SGD stability  ✅ proved (conditional, honest)

`T4_curvature_aware_cap`.

If (i) `Dperp ⟂ g`, (ii) the directions are eigen-aligned so the cross curvature
term vanishes (`Dparᵀ H Dperp = 0`), (iii) SGD's aligned part strictly descends,
and (iv) the controller uses **curvature information** to cap the transverse
energy below the aligned descent it achieved (`½ Dperpᵀ H Dperp ≤ −ΔL_sgd`),
then the full momentum step also descends (`ΔL_mom ≤ 0`).

Assumptions stated honestly: the cap needs `Dperpᵀ H Dperp`, a Hessian
(curvature) quantity — exactly the information a geometry-blind cap (T3) lacks;
and eigen-alignment is what makes the cross term drop. The theorem is a
conditional safety criterion, not an unconditional win: it says the *right* extra
signal (curvature) is sufficient to restore the descent guarantee that a scalar
cap cannot.

---

## Proof status summary

| Theorem | Lean name | Status |
|---|---|---|
| T1 exact descent-gap identity | `T1_descent_gap` | proved, sorry-free |
| T1 matched-aligned instantiation | `T1_matched_sgd` | proved, sorry-free |
| T2 SGD descent (matched, ℓ_y-free) | `T2_sgd_descends` | proved, sorry-free |
| T2 anisotropic ascent | `T2_aniso_mom_ascends_pos` | proved, sorry-free |
| T2 isotropic descent (benign) | `T2_iso_mom_descends_neg` | proved, sorry-free |
| T2 exact threshold ℓ_y⋆ = 33.11 | `T2_threshold` | proved, sorry-free |
| T3 scalar-optimality of bestReduction | `bestReduction_is_scalar_optimum` | proved, sorry-free |
| T3 strict gap vs tuned SGD | `T3_strict_gap` | proved, sorry-free |
| T3 no uniform bound (impossibility) | `T3_no_uniform_bound` | proved, sorry-free |
| T4 curvature-aware cap preserves descent | `T4_curvature_aware_cap` | proved, sorry-free |

All ten depend only on `[propext, Classical.choice, Quot.sound]`.

## Concrete counterexample instances (verified in Lean as exact rationals)

- **T1/T2 poison instance** (numerically pre-searched in `.venv` python, then
  formalized): `H = diag(1,100)`, `θ = g = (1,0)`, buffer `b = (0,1)`,
  `μ = 9/10`, `η_m = 3/20` (so `η_s = 57/200 ≈ 0.28`). Matched SGD descends
  `0.2444`; momentum ascends `+0.4937`. The same buffer/step at `H = diag(1,1)`
  descends `−0.2370` (harm `0.0074`). Threshold `ℓ_y⋆ = 3910200/118098 ≈ 33.11`.
- **T3 impossibility witness**: at `lx=gx=by=1, μ=9/10`, `ℓ_y=100` gives ratio
  `1/82 ≈ 0.0122` — the geometry-blind controller, even at its own optimal
  scalar, gets ~1.2% of tuned SGD's one-step descent; `→ 0` as `ℓ_y → ∞`.

## Product implication (per the user's framing)

The next-generation outer optimizer must **either drop directional (first-moment)
memory entirely OR acquire curvature/covariance information**. A controller that
sees only gain/cosine of `(g_t, b_{t-1})` and outputs a scalar step-scale is
provably insufficient: it cannot cancel the transverse contamination `Dperp`
(T1), that contamination is catastrophic under the anisotropic curvature this
regime exhibits (T2), and no such scalar dominates tuned SGD across the quadratic
family (T3). Restoring the descent guarantee provably requires curvature input
(T4). This is the machine-checked backbone of docs/NEXT_OPTIMIZER_PLAN.md's
decision to build on the memoryless SGD base and add only current-round
spatial/curvature/consensus signals — never a directional buffer.

Scope caveats (honest): 2D and per-step (no multi-step/convergence claim);
exact-arithmetic (no f32 rounding, cf. THEORY.md §0); `g_t = ∇L` in the model
(the production pseudo-gradient-vs-gradient gap is THEORY.md Conjecture C.5); T4
is a conditional (eigen-aligned, curvature-cap) sufficiency statement.
