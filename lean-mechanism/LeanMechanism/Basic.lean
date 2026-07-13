import Mathlib

/-!
# The DiLoCo transverse-momentum mechanism, on a minimal 2D quadratic

This file machine-checks the four theorems that compress the DiLoCo-poison
empirical findings (docs/THEORY.md Result A/B, docs/EXP2_37.md matched-η_eff
triads) onto a minimal anisotropic 2D quadratic model.

## Model

`L(p) = ½ pᵀ H p` with `H` a symmetric 2×2 matrix, entries `a = H₁₁`,
`b = H₂₂`, `c = H₁₂ = H₂₁`.  The gradient is `∇L(θ) = H θ`.  The one-step
loss change of applying a displacement `u` (i.e. `θ ↦ θ − u`) on a quadratic
is *exact* (no smoothness slack):

    ΔL(u) = L(θ − u) − L(θ) = − uᵀ(Hθ) + ½ uᵀ H u.

The pseudo-gradient is `g_t = ∇L(θ) = Hθ` (minimal model: even when the merged
delta equals the true gradient, the *buffer* carries a foreign direction).

* SGD applies      `u_sgd  = η_s · g_t`.
* Nesterov applies `u_mom  = η_m · d_t`, `d_t = (1+μ) g_t + μ² b_{t-1}`,
  which splits as `d_t = A g_t + d⊥`, `A = 1+μ+μ² c_t`, `d⊥ = μ² b⊥ ⟂ g_t`.

We write the two applied displacements as a common aligned part `Dpar ∥ g_t`
(the "matched aligned gain": `η_s` is chosen so `u_sgd = Dpar = η_m A g_t`)
plus, for momentum only, a transverse part `Dperp ⟂ g_t`.
-/

namespace LeanMechanism

noncomputable section

/-- `uᵀ H u`, the quadratic form of the symmetric 2×2 `H = [[a,c],[c,b]]`. -/
def Q (a b c : ℝ) (u : ℝ × ℝ) : ℝ := a * u.1 ^ 2 + 2 * c * u.1 * u.2 + b * u.2 ^ 2

/-- `uᵀ H w`, the symmetric bilinear form of `H = [[a,c],[c,b]]`. -/
def Bil (a b c : ℝ) (u w : ℝ × ℝ) : ℝ :=
  a * u.1 * w.1 + c * (u.1 * w.2 + u.2 * w.1) + b * u.2 * w.2

/-- `H v`, matrix-vector product for the symmetric 2×2 `H = [[a,c],[c,b]]`. -/
def Hmul (a b c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1 + c * v.2, c * v.1 + b * v.2)

/-- The quadratic loss `L(p) = ½ pᵀ H p`. -/
def Lq (a b c : ℝ) (p : ℝ × ℝ) : ℝ := (1 / 2) * Q a b c p

/-- One-step loss change of the update `θ ↦ θ − u`, exact for the quadratic. -/
def dL (a b c : ℝ) (θ u : ℝ × ℝ) : ℝ := Lq a b c (θ.1 - u.1, θ.2 - u.2) - Lq a b c θ

/-- Euclidean inner product on `ℝ²`. -/
def dot (u w : ℝ × ℝ) : ℝ := u.1 * w.1 + u.2 * w.2

/-! ## T1 — Matched aligned gain does not imply matched descent -/

/-- **T1 (exact decomposition of the descent gap).**
Suppose the pseudo-gradient is the true gradient `g = Hθ`, SGD applies the
aligned displacement `Dpar` and momentum applies `Dpar + Dperp` with the extra
part transverse to `g` (`⟨Dperp, g⟩ = 0`).  Then the *difference* of the two
one-step loss changes is exactly

    ΔL_mom − ΔL_sgd = Dparᵀ H Dperp + ½ Dperpᵀ H Dperp
                    = ½ ( Dperpᵀ H Dperp + 2 Dparᵀ H Dperp ).

Matching the aligned projection (equal `Dpar`) therefore does *not* equalize
descent: the gap is precisely the transverse curvature energy plus the cross
curvature term — the sharp-direction contamination the aligned match cannot see.
It is nonzero whenever `Dperpᵀ H Dperp > 0` and the cross term does not cancel
it.  (The identity is exact for any shared `Dpar`; the "matched aligned gain"
setup fixes `Dpar = η_m A g_t = u_sgd`, formalized in `T1_matched_sgd` below.) -/
theorem T1_descent_gap (a b c : ℝ) (θ Dpar Dperp : ℝ × ℝ)
    (hperp : dot Dperp (Hmul a b c θ) = 0) :
    dL a b c θ (Dpar.1 + Dperp.1, Dpar.2 + Dperp.2) - dL a b c θ Dpar
      = Bil a b c Dpar Dperp + (1 / 2) * Q a b c Dperp := by
  obtain ⟨θ1, θ2⟩ := θ
  obtain ⟨p1, p2⟩ := Dpar
  obtain ⟨q1, q2⟩ := Dperp
  simp only [dL, Lq, Q, Bil, Hmul, dot] at *
  linear_combination -hperp

/-- **T1 (matched-aligned instantiation).** Nesterov with a buffer `buf`
transverse to the gradient `g = Hθ` (`⟨buf, g⟩ = 0`, i.e. realized geometry
`c_t = 0`, aligned gain `A = 1+μ`) applies `umom = η_m·((1+μ)g + μ² buf)`; SGD at
the *matched* LR `η_s = η_m·(1+μ)` applies `usgd = η_m(1+μ)·g`. Then (i) the two
steps share the aligned part exactly (`umom = usgd + Dperp`, `Dperp = η_m μ² buf`
purely transverse), yet (ii) their one-step loss changes differ by the exact
transverse form of T1.  Matched aligned gain, unequal descent. -/
theorem T1_matched_sgd (a b c mu etam : ℝ) (θ buf : ℝ × ℝ)
    (hbuf : dot buf (Hmul a b c θ) = 0) :
    let g := Hmul a b c θ
    let usgd := (etam * (1 + mu) * g.1, etam * (1 + mu) * g.2)
    let umom := (etam * ((1 + mu) * g.1 + mu ^ 2 * buf.1),
                 etam * ((1 + mu) * g.2 + mu ^ 2 * buf.2))
    let Dperp := (etam * mu ^ 2 * buf.1, etam * mu ^ 2 * buf.2)
    umom = (usgd.1 + Dperp.1, usgd.2 + Dperp.2) ∧
      dot Dperp g = 0 ∧
        dL a b c θ umom - dL a b c θ usgd
          = Bil a b c usgd Dperp + (1 / 2) * Q a b c Dperp := by
  intro g usgd umom Dperp
  have hsum : umom = (usgd.1 + Dperp.1, usgd.2 + Dperp.2) := by
    simp only [umom, usgd, Dperp, Prod.mk.injEq]; constructor <;> ring
  have hperp : dot Dperp g = 0 := by
    obtain ⟨θ1, θ2⟩ := θ; obtain ⟨b1, b2⟩ := buf
    simp only [Dperp, g, dot, Hmul] at *
    linear_combination (etam * mu ^ 2) * hbuf
  refine ⟨hsum, hperp, ?_⟩
  rw [hsum]
  exact T1_descent_gap a b c θ usgd Dperp hperp

/-! ## T2 — Transverse momentum: benign under isotropy, harmful under anisotropy

Concrete verified instance (matched aligned step `η_eff = η_s ≈ 0.28`, the
tuned-SGD LR; `μ = 0.9`, buffer purely in the sharp direction):

  H = diag(1, ℓ_y),  θ = (1,0),  g = (1,0),  b = (0,1),
  Dpar = (57/200, 0) = η_m A g   (η_m = 3/20, A = 1+μ = 19/10; so η_s = 57/200),
  Dperp = (0, 243/2000) = η_m μ² b.

At `ℓ_y = 100` (anisotropic: sharp direction 100× the flat curvature that carries
the gradient — the regime where the empirical matched-η_eff harm is severe)
momentum turns a descent into an ascent; at `ℓ_y = 1` (isotropic) momentum still
descends and the harm is negligible. Same buffer, same aligned step — only the
curvature changes. -/

/-- Matched-aligned SGD strictly descends (identical aligned part in both
curvatures), reduction `≈ 0.2444`. -/
theorem T2_sgd_descends (ly : ℝ) :
    dL 1 ly 0 (1, 0) (57 / 200, 0) = -19551 / 80000 := by
  simp only [dL, Lq, Q]; norm_num

/-- **T2, anisotropic (ℓ_y = 100): momentum strictly INCREASES one-step loss**
(`ΔL_mom = +0.4937 > 0`) while matched SGD descends by `0.2444`. -/
theorem T2_aniso_mom_ascends :
    dL 1 100 0 (1, 0) (57 / 200 + 0, 0 + 243 / 2000) = 1974900 / 4000000 := by
  simp only [dL, Lq, Q]; norm_num

theorem T2_aniso_mom_ascends_pos :
    0 < dL 1 100 0 (1, 0) (57 / 200 + 0, 0 + 243 / 2000) := by
  rw [T2_aniso_mom_ascends]; norm_num

/-- **T2, isotropic (ℓ_y = 1): momentum still DESCENDS** (`ΔL_mom = −0.2370 < 0`),
the transverse harm is a negligible `0.0074` on top of the `0.2444` SGD descent. -/
theorem T2_iso_mom_descends :
    dL 1 1 0 (1, 0) (57 / 200 + 0, 0 + 243 / 2000) = -237006375 / 1000000000 := by
  simp only [dL, Lq, Q]; norm_num

theorem T2_iso_mom_descends_neg :
    dL 1 1 0 (1, 0) (57 / 200 + 0, 0 + 243 / 2000) < 0 := by
  rw [T2_iso_mom_descends]; norm_num

/-- **T2 threshold (exact).** With the concrete `Dpar, Dperp` above and diagonal
`H = diag(1, ℓ_y)`, momentum flips from net descent to net ascent exactly at the
transverse curvature `ℓ_y⋆ = 3910200/118098 ≈ 33.11`:
`ΔL_mom(ℓ_y) > 0 ↔ ℓ_y > ℓ_y⋆`. -/
theorem T2_threshold (ly : ℝ) :
    0 < dL 1 ly 0 (1, 0) (57 / 200 + 0, 0 + 243 / 2000) ↔ 3910200 / 118098 < ly := by
  simp only [dL, Lq, Q]
  constructor
  · intro h; nlinarith [h]
  · intro h; nlinarith [h]

/-! ## T3 — No geometry-blind scalar cap dominates tuned SGD over all quadratics

A controller that applies a scalar step-scale `s(g,b)` to `d = g + μb` is stuck
with the *direction* `d`; only its length is free. Even optimizing that scalar,
its best one-step reduction along `d` is `R(d) = ⟨d,g⟩² / (2 dᵀHd)`, versus tuned
SGD's `R(g) = ⟨g,g⟩² / (2 gᵀHg)`.  For diagonal `H = diag(ℓx, ℓy)`, `g = (gx,0)`,
`b = (0,by)` (buffer transverse, sharp-direction mass), the ratio is
`ℓx gx² / (ℓx gx² + μ² ℓy by²) < 1`, driven to `0` by increasing the sharp
curvature `ℓy` — which the controller cannot observe (`g, b`, hence `s`, fixed). -/

/-- Best one-step loss reduction achievable moving along direction `v` with an
optimal scalar step, on the quadratic `H = diag(lx, ly)`, gradient `g`:
`R = ⟨v,g⟩² / (2 vᵀHv)`. -/
def bestReduction (lx ly : ℝ) (v g : ℝ × ℝ) : ℝ :=
  (dot v g) ^ 2 / (2 * (lx * v.1 ^ 2 + ly * v.2 ^ 2))

/-- The one-step loss reduction `−ΔL` of a *scalar* step `s·v` along `v` on the
diagonal quadratic is the 1-D quadratic `s ⟨v,g⟩ − ½ s² vᵀHv`, `g = Hθ`. -/
theorem neg_dL_scalar (lx ly : ℝ) (θ v : ℝ × ℝ) (s : ℝ) :
    - dL lx ly 0 θ (s * v.1, s * v.2)
      = s * dot v (Hmul lx ly 0 θ) - (1 / 2) * s ^ 2 * (lx * v.1 ^ 2 + ly * v.2 ^ 2) := by
  obtain ⟨θ1, θ2⟩ := θ; obtain ⟨v1, v2⟩ := v
  simp only [dL, Lq, Q, dot, Hmul]; ring

/-- **`bestReduction` really is the scalar optimum of the loss.** For any
direction `v` with positive curvature `vᵀHv > 0`, no scalar step beats
`bestReduction`, and the optimal scalar `s⋆ = ⟨v,g⟩ / vᵀHv` attains it. This is
what makes T3 a statement about the *actual one-step loss*, not just a chosen
formula: a scalar controller restricted to the direction `v = d = g + μb` can do
no better than `bestReduction lx ly d g`. -/
theorem bestReduction_is_scalar_optimum (lx ly : ℝ) (θ v : ℝ × ℝ)
    (hQ : 0 < lx * v.1 ^ 2 + ly * v.2 ^ 2) :
    (∀ s : ℝ, - dL lx ly 0 θ (s * v.1, s * v.2) ≤ bestReduction lx ly v (Hmul lx ly 0 θ)) ∧
      (- dL lx ly 0 θ
            ((dot v (Hmul lx ly 0 θ) / (lx * v.1 ^ 2 + ly * v.2 ^ 2)) * v.1,
             (dot v (Hmul lx ly 0 θ) / (lx * v.1 ^ 2 + ly * v.2 ^ 2)) * v.2)
          = bestReduction lx ly v (Hmul lx ly 0 θ)) := by
  set D := dot v (Hmul lx ly 0 θ) with hD
  set Qv := lx * v.1 ^ 2 + ly * v.2 ^ 2 with hQv
  have hbr : bestReduction lx ly v (Hmul lx ly 0 θ) = D ^ 2 / (2 * Qv) := by
    rw [bestReduction, hD, hQv]
  refine ⟨fun s => ?_, ?_⟩
  · rw [neg_dL_scalar, hbr, le_div_iff₀ (by positivity)]
    nlinarith [sq_nonneg (D - s * Qv)]
  · simp only [neg_dL_scalar, hbr]
    field_simp
    ring

/-- **T3 (per-instance strict gap).** For diagonal `H`, gradient `g=(gx,0)`,
buffer `b=(0,by)`, momentum direction `d=(gx, μ·by)`: the controller's best
reduction along `d` is *strictly less* than tuned SGD's along `g`, whenever the
transverse sharp mass is nonzero. -/
theorem T3_strict_gap (lx ly gx by_ mu : ℝ)
    (hlx : 0 < lx) (hly : 0 < ly) (hgx : gx ≠ 0) (hby : by_ ≠ 0) (hmu : mu ≠ 0) :
    bestReduction lx ly (gx, mu * by_) (gx, 0) < bestReduction lx ly (gx, 0) (gx, 0) := by
  have hgx2 : 0 < gx ^ 2 := by positivity
  have hnum : 0 < (gx ^ 2) ^ 2 := by positivity
  unfold bestReduction dot
  have e1 : ((gx, mu * by_).1 * (gx, (0:ℝ)).1 + (gx, mu * by_).2 * (gx, (0:ℝ)).2) = gx ^ 2 := by
    dsimp only; try ring
  have e2 : ((gx, (0:ℝ)).1 * (gx, (0:ℝ)).1 + (gx, (0:ℝ)).2 * (gx, (0:ℝ)).2) = gx ^ 2 := by
    dsimp only; try ring
  have d2 : (2 : ℝ) * (lx * (gx, (0:ℝ)).1 ^ 2 + ly * (gx, (0:ℝ)).2 ^ 2) = 2 * (lx * gx ^ 2) := by
    dsimp only; try ring
  have d3 : (2 : ℝ) * (lx * (gx, mu * by_).1 ^ 2 + ly * (gx, mu * by_).2 ^ 2)
      = 2 * (lx * gx ^ 2 + ly * (mu * by_) ^ 2) := by dsimp only; try ring
  rw [e1, e2, d2, d3]
  apply div_lt_div_of_pos_left hnum (by positivity)
  nlinarith [mul_pos hly (by positivity : (0:ℝ) < (mu * by_) ^ 2)]

/-- **T3 (no uniform bound / impossibility).** For every target ratio `κ ∈ (0,1)`
there is an anisotropic quadratic (a large enough sharp curvature `ℓy`) on which
the geometry-blind controller's best reduction is below `κ ·` tuned SGD's. Since
`gx, by, μ` (hence any scalar `s`) are fixed while `ℓy` varies, no fixed scalar
controller can stay within any constant factor of tuned SGD across the family. -/
theorem T3_no_uniform_bound (gx by_ mu : ℝ)
    (hgx : gx ≠ 0) (hby : by_ ≠ 0) (hmu : mu ≠ 0) (κ : ℝ) (hκ0 : 0 < κ) (hκ1 : κ < 1) :
    ∃ ly : ℝ, 0 < ly ∧
      bestReduction 1 ly (gx, mu * by_) (gx, 0) < κ * bestReduction 1 ly (gx, 0) (gx, 0) := by
  have hgx2 : 0 < gx ^ 2 := by positivity
  have hby2 : 0 < by_ ^ 2 := by positivity
  have hmu2 : 0 < mu ^ 2 := by positivity
  refine ⟨(gx ^ 2 * (1 - κ)) / (κ * mu ^ 2 * by_ ^ 2) + 1, by positivity, ?_⟩
  set ly := (gx ^ 2 * (1 - κ)) / (κ * mu ^ 2 * by_ ^ 2) + 1 with hlydef
  have hly : 0 < ly := by positivity
  have hkey : gx ^ 2 * (1 - κ) < κ * (ly * (mu ^ 2 * by_ ^ 2)) := by
    have hden : 0 < κ * mu ^ 2 * by_ ^ 2 := by positivity
    have hexp : κ * (ly * (mu ^ 2 * by_ ^ 2))
        = gx ^ 2 * (1 - κ) + κ * mu ^ 2 * by_ ^ 2 := by
      rw [hlydef]; field_simp; try ring
    nlinarith [hexp, hden]
  unfold bestReduction dot
  have e1 : ((gx, mu * by_).1 * (gx, (0:ℝ)).1 + (gx, mu * by_).2 * (gx, (0:ℝ)).2) = gx ^ 2 := by
    dsimp only; try ring
  have e2 : ((gx, (0:ℝ)).1 * (gx, (0:ℝ)).1 + (gx, (0:ℝ)).2 * (gx, (0:ℝ)).2) = gx ^ 2 := by
    dsimp only; try ring
  have d2 : (2 : ℝ) * (1 * (gx, (0:ℝ)).1 ^ 2 + ly * (gx, (0:ℝ)).2 ^ 2) = 2 * gx ^ 2 := by
    dsimp only; try ring
  have d3 : (2 : ℝ) * (1 * (gx, mu * by_).1 ^ 2 + ly * (gx, mu * by_).2 ^ 2)
      = 2 * (gx ^ 2 + ly * (mu ^ 2 * by_ ^ 2)) := by dsimp only; try ring
  rw [e1, e2, d2, d3]
  have hd1 : (0:ℝ) < 2 * (gx ^ 2 + ly * (mu ^ 2 * by_ ^ 2)) := by positivity
  rw [div_lt_iff₀ hd1]
  have hrhs : κ * ((gx ^ 2) ^ 2 / (2 * gx ^ 2)) * (2 * (gx ^ 2 + ly * (mu ^ 2 * by_ ^ 2)))
      = κ * (gx ^ 2 * (gx ^ 2 + ly * (mu ^ 2 * by_ ^ 2))) := by
    field_simp; try ring
  rw [hrhs]
  nlinarith [mul_lt_mul_of_pos_left hkey hgx2, hgx2]

/-! ## T4 — A curvature-aware transverse cap preserves SGD stability -/

/-- **T4 (curvature-aware cap preserves descent).** Suppose the pseudo-gradient
is the true gradient `g = Hθ`, the aligned (SGD) part `Dpar` strictly descends,
the directions are eigen-aligned so the cross curvature term vanishes
(`Dparᵀ H Dperp = 0`), and the controller uses *curvature information* to cap the
transverse energy below the aligned descent it achieved
(`½ Dperpᵀ H Dperp ≤ −ΔL_sgd`).  Then the full momentum step also descends
(`ΔL_mom ≤ 0`).  The cap needs `Dperpᵀ H Dperp` (a Hessian quantity), and
eigen-alignment is what makes the cross term drop — exactly the information a
geometry-blind cap (T3) lacks. -/
theorem T4_curvature_aware_cap (a b c : ℝ) (θ Dpar Dperp : ℝ × ℝ)
    (hperp : dot Dperp (Hmul a b c θ) = 0)
    (hcross : Bil a b c Dpar Dperp = 0)
    (hsgd : dL a b c θ Dpar < 0)
    (hcap : (1 / 2) * Q a b c Dperp ≤ - dL a b c θ Dpar) :
    dL a b c θ (Dpar.1 + Dperp.1, Dpar.2 + Dperp.2) ≤ 0 := by
  have hid := T1_descent_gap a b c θ Dpar Dperp hperp
  rw [hcross, zero_add] at hid
  linarith [hid, hcap, hsgd]

end

end LeanMechanism
