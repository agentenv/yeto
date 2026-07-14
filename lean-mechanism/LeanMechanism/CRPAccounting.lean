import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# CRP residual accounting

This file formalizes only the exact Euclidean accounting identities needed to
reason about a residual correction.  It also gives a concrete counterexample
showing that vector norms do not identify the missing cross term.  These
statements prove no optimizer superiority, convergence, or loss improvement.
-/

namespace LeanMechanism

noncomputable section

open scoped BigOperators InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E]

/-- Signed CRP residual under the executable policy convention: proposal minus stock. -/
def crpResidual (G Q : E) : E :=
  Q - G

/-- The residual is exactly proposal minus stock. -/
theorem crpResidual_identity (G Q : E) : crpResidual G Q = Q - G :=
  rfl

/-- Adding the full residual to stock recovers the proposal. -/
theorem stock_add_crpResidual (G Q : E) : G + crpResidual G Q = Q := by
  simp [crpResidual]

variable [InnerProductSpace ℝ E]

/-- Exact squared-norm expansion for a scalar residual correction.  The
middle inner-product term is the cross term that separate norms do not retain. -/
theorem norm_sq_stock_sub_residual (G r : E) (lambda : ℝ) :
    ‖G - lambda • r‖ ^ 2 =
      ‖G‖ ^ 2 - 2 * lambda * ⟪G, r⟫_ℝ + lambda ^ 2 * ‖r‖ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq]
  simp only [inner_sub_left, inner_sub_right, real_inner_smul_left,
    real_inner_smul_right]
  rw [real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq,
    real_inner_comm r G]
  ring

/-- Exact squared-norm expansion for the executable additive residual pulse. -/
theorem norm_sq_stock_add_residual (G r : E) (lambda : ℝ) :
    ‖G + lambda • r‖ ^ 2 =
      ‖G‖ ^ 2 + 2 * lambda * ⟪G, r⟫_ℝ + lambda ^ 2 * ‖r‖ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq]
  simp only [inner_add_left, inner_add_right, real_inner_smul_left,
    real_inner_smul_right]
  rw [real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq,
    real_inner_comm r G]
  ring

/-- The additive expansion specialized to the signed CRP residual `Q - G`. -/
theorem norm_sq_stock_add_crpResidual (G Q : E) (lambda : ℝ) :
    ‖G + lambda • crpResidual G Q‖ ^ 2 =
      ‖G‖ ^ 2 + 2 * lambda * ⟪G, crpResidual G Q⟫_ℝ +
        lambda ^ 2 * ‖crpResidual G Q‖ ^ 2 :=
  norm_sq_stock_add_residual G (crpResidual G Q) lambda

/-- Coordinate-wise squared energy for a finite real vector. -/
def coordinateEnergy {n : ℕ} (x : Fin n → ℝ) : ℝ :=
  ∑ i, (x i) ^ 2

/-- Coordinate-wise cross term for two finite real vectors. -/
def coordinateCross {n : ℕ} (x y : Fin n → ℝ) : ℝ :=
  ∑ i, x i * y i

/-- Residual formation is component-wise subtraction. -/
theorem crpResidual_apply {n : ℕ} (G Q : Fin n → ℝ) (i : Fin n) :
    crpResidual G Q i = Q i - G i :=
  rfl

/-- Finite-coordinate accounting for a scalar residual correction.  The total
energy is the sum of the stock energy, cross term, and residual energy. -/
theorem coordinateEnergy_stock_sub_residual {n : ℕ}
    (G r : Fin n → ℝ) (lambda : ℝ) :
    coordinateEnergy (fun i ↦ G i - lambda * r i) =
      coordinateEnergy G - 2 * lambda * coordinateCross G r +
        lambda ^ 2 * coordinateEnergy r := by
  unfold coordinateEnergy coordinateCross
  calc
    ∑ i, (G i - lambda * r i) ^ 2 =
        ∑ i, ((G i) ^ 2 - (2 * lambda) * (G i * r i) +
          lambda ^ 2 * (r i) ^ 2) := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = (∑ i, (G i) ^ 2) - 2 * lambda * (∑ i, G i * r i) +
        lambda ^ 2 * (∑ i, (r i) ^ 2) := by
      rw [Finset.sum_add_distrib, Finset.sum_sub_distrib,
        ← Finset.mul_sum, ← Finset.mul_sum]

/-- The stock/residual cross term itself requires component-level stock and
proposal interaction; it is the stock/proposal cross term minus stock energy. -/
theorem coordinateCross_crpResidual {n : ℕ} (G Q : Fin n → ℝ) :
    coordinateCross G (crpResidual G Q) =
      coordinateCross G Q - coordinateEnergy G := by
  unfold coordinateCross coordinateEnergy crpResidual
  calc
    ∑ i, G i * (Q - G) i = ∑ i, (G i * Q i - (G i) ^ 2) := by
      apply Finset.sum_congr rfl
      intro i _
      simp only [Pi.sub_apply]
      ring
    _ = (∑ i, G i * Q i) - ∑ i, (G i) ^ 2 := by
      rw [Finset.sum_sub_distrib]

/-- Concrete one-dimensional counterexample: the stock and two residuals have
the same norms, but their cross terms with stock differ.  Therefore norms alone
cannot identify the cross term in `norm_sq_stock_sub_residual`. -/
theorem norms_alone_do_not_identify_cross_term :
    ∃ G r₁ r₂ : ℝ,
      ‖G‖ = 1 ∧ ‖r₁‖ = 1 ∧ ‖r₂‖ = 1 ∧
        ⟪G, r₁⟫_ℝ ≠ ⟪G, r₂⟫_ℝ := by
  refine ⟨1, 1, -1, ?_⟩
  norm_num [Real.norm_eq_abs]

/-- The same norms-only counterexample stated at the resulting corrected
energy: identical stock/residual norms can yield different candidate norms. -/
theorem norms_alone_do_not_identify_corrected_energy :
    ∃ G r₁ r₂ lambda : ℝ,
      ‖G‖ = 1 ∧ ‖r₁‖ = 1 ∧ ‖r₂‖ = 1 ∧
        ‖G - lambda • r₁‖ ^ 2 ≠ ‖G - lambda • r₂‖ ^ 2 := by
  refine ⟨1, 1, -1, 1, ?_⟩
  norm_num [Real.norm_eq_abs]

end

end LeanMechanism
