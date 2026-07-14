import LeanMechanism.CRPAccounting

/-!
Direct compile/audit target for the CRP accounting layer.  The examples check
the public theorem interfaces, and `#print axioms` exposes their complete axiom
dependencies to the build log.
-/

open scoped InnerProductSpace

namespace LeanMechanism.Tests

example {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (G r : E) (lambda : ℝ) :
    ‖G - lambda • r‖ ^ 2 =
      ‖G‖ ^ 2 - 2 * lambda * ⟪G, r⟫_ℝ + lambda ^ 2 * ‖r‖ ^ 2 :=
  norm_sq_stock_sub_residual G r lambda

example {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (G r : E) (lambda : ℝ) :
    ‖G + lambda • r‖ ^ 2 =
      ‖G‖ ^ 2 + 2 * lambda * ⟪G, r⟫_ℝ + lambda ^ 2 * ‖r‖ ^ 2 :=
  norm_sq_stock_add_residual G r lambda

example {n : ℕ} (G r : Fin n → ℝ) (lambda : ℝ) :
    coordinateEnergy (fun i ↦ G i - lambda * r i) =
      coordinateEnergy G - 2 * lambda * coordinateCross G r +
        lambda ^ 2 * coordinateEnergy r :=
  coordinateEnergy_stock_sub_residual G r lambda

example :
    ∃ G r₁ r₂ : ℝ,
      ‖G‖ = 1 ∧ ‖r₁‖ = 1 ∧ ‖r₂‖ = 1 ∧
        ⟪G, r₁⟫_ℝ ≠ ⟪G, r₂⟫_ℝ :=
  norms_alone_do_not_identify_cross_term

#print axioms LeanMechanism.crpResidual_identity
#print axioms LeanMechanism.stock_add_crpResidual
#print axioms LeanMechanism.norm_sq_stock_sub_residual
#print axioms LeanMechanism.norm_sq_stock_add_residual
#print axioms LeanMechanism.norm_sq_stock_add_crpResidual
#print axioms LeanMechanism.crpResidual_apply
#print axioms LeanMechanism.coordinateEnergy_stock_sub_residual
#print axioms LeanMechanism.coordinateCross_crpResidual
#print axioms LeanMechanism.norms_alone_do_not_identify_cross_term
#print axioms LeanMechanism.norms_alone_do_not_identify_corrected_energy

end LeanMechanism.Tests
