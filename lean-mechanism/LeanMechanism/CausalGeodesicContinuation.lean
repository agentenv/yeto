import LeanMechanism.PrequentialTransverseInterlock

/-!
# Causal geodesic-continuation geometry

This file proves only exact geometry for a bounded, norm-grafted continuation
away from the previous direction.  It proves no stochastic convergence,
language-model loss improvement, or empirical superiority over SGD.
-/

namespace LeanMechanism

noncomputable section

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- CGC turns away from the backward tangent `b` by tangent ratio `lambda`. -/
def cgcCandidate (u b : E) (lambda : ℝ) : E :=
  transverseCandidate u b (-lambda)

/-- Under the orthonormal hypotheses, CGC preserves unit direction norm. -/
theorem norm_cgcCandidate (u b : E) (lambda : ℝ)
    (hu : ‖u‖ = 1) (hb : ‖b‖ = 1) (hub : ⟪u, b⟫_ℝ = 0) :
    ‖cgcCandidate u b lambda‖ = 1 := by
  exact norm_transverseCandidate u b (-lambda) hu hb hub

/-- Zero observed angular velocity degenerates exactly to the stock direction. -/
theorem cgcCandidate_zero (u b : E) : cgcCandidate u b 0 = u := by
  simp [cgcCandidate, transverseCandidate, transverseScale]

/-- For positive unit-circle components, the uncapped tangent ratio `s / rho`
continues the observed turn exactly: the backward component changes sign. -/
theorem cgcCandidate_exact_continuation (u b : E) (rho s : ℝ)
    (hrho : 0 < rho) (hunit : rho ^ 2 + s ^ 2 = 1) :
    cgcCandidate u b (s / rho) = rho • u - s • b := by
  have hrho_ne : rho ≠ 0 := ne_of_gt hrho
  have hone_div_nonneg : 0 ≤ (1 / rho : ℝ) := (one_div_pos.mpr hrho).le
  have harg : 1 + (s / rho) ^ 2 = (1 / rho) ^ 2 := by
    field_simp
    nlinarith
  have hscale : transverseScale (s / rho) = 1 / rho := by
    rw [transverseScale, harg, Real.sqrt_sq hone_div_nonneg]
  have hscale_neg : transverseScale (-(s / rho)) = 1 / rho := by
    simpa [transverseScale] using hscale
  have hcoef : rho * (s * rho⁻¹) = s := by
    field_simp
  unfold cgcCandidate transverseCandidate
  rw [hscale_neg]
  simp [div_eq_mul_inv, smul_add, smul_smul, sub_eq_add_neg, hcoef]

/-- The current-direction cosine is exactly the inverse normalization factor. -/
theorem inner_cgcCandidate_current (u b : E) (lambda : ℝ)
    (hu : ‖u‖ = 1) (hub : ⟪u, b⟫_ℝ = 0) :
    ⟪cgcCandidate u b lambda, u⟫_ℝ = 1 / transverseScale lambda := by
  have hbu : ⟪b, u⟫_ℝ = 0 := by
    rw [real_inner_comm, hub]
  unfold cgcCandidate transverseCandidate
  rw [real_inner_smul_left, inner_add_left, real_inner_smul_left,
    real_inner_self_eq_norm_sq, hu, hbu]
  simp [transverseScale]

/-- A cap `|lambda| ≤ 1/4` bounds the squared cosine with the stock direction
below by `16/17`; this is the algebraic form of the `atan(1/4)` angle cap. -/
theorem cgcCandidate_cap_cosine_sq (u b : E) (lambda : ℝ)
    (hu : ‖u‖ = 1) (hub : ⟪u, b⟫_ℝ = 0)
    (hlower : -(1 / 4 : ℝ) ≤ lambda) (hupper : lambda ≤ (1 / 4 : ℝ)) :
    (16 / 17 : ℝ) ≤ ⟪cgcCandidate u b lambda, u⟫_ℝ ^ 2 := by
  rw [inner_cgcCandidate_current u b lambda hu hub]
  have hscale_sq : transverseScale lambda ^ 2 = 1 + lambda ^ 2 := by
    rw [transverseScale, Real.sq_sqrt]
    nlinarith [sq_nonneg lambda]
  have hlambda_sq : lambda ^ 2 ≤ (1 / 16 : ℝ) := by
    nlinarith
  have hden_pos : 0 < 1 + lambda ^ 2 := by nlinarith [sq_nonneg lambda]
  rw [one_div, inv_pow, hscale_sq]
  rw [inv_eq_one_div]
  apply (le_div_iff₀ hden_pos).2
  nlinarith

/-- If the next direction reverses toward the old backward tangent, a positive
CGC continuation is strictly worse in alignment than stock.  This explicit
counterexample pattern rules out unconditional dominance. -/
theorem cgc_reversal_alignment_lt (u b : E) (lambda : ℝ)
    (hub : ⟪u, b⟫_ℝ = 0) (hb : ‖b‖ = 1) (hlambda : 0 < lambda) :
    ⟪cgcCandidate u b lambda, b⟫_ℝ < ⟪u, b⟫_ℝ := by
  have hscale_pos := transverseScale_pos lambda
  have hbb : ⟪b, b⟫_ℝ = 1 := by
    rw [real_inner_self_eq_norm_sq, hb]
    norm_num
  have hscale_neg : transverseScale (-lambda) = transverseScale lambda := by
    simp [transverseScale]
  unfold cgcCandidate transverseCandidate
  rw [real_inner_smul_left, inner_add_left, real_inner_smul_left, hub, hbb,
    hscale_neg]
  simp only [zero_add, mul_one]
  exact mul_neg_of_pos_of_neg (one_div_pos.mpr hscale_pos) (neg_neg_of_pos hlambda)

end

end LeanMechanism
