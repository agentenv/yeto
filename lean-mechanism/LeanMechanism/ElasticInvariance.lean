import Mathlib
import LeanMechanism.QuadraticAlignment

/-!
# Elastic-cadence invariance

A cadence is represented by a list of positive inner-step counts.  Its sum is
the total inner-step budget and its length is the number of outer updates.

The theorem here has deliberately narrow modeling scope.  It uses the same
constant-pseudo-input, one-dimensional frozen-gradient model as
`QuadraticAlignment.lean`.  In that model, bias correction makes every outer
update carry the same multiplier `1/(1-mu)`, so a cadence affects the
accumulated multiplier and the corrected optimum only through the number `T` of
outer updates.  The theorem does **not** say that changing real wall-clock or
inner-training cadence leaves the pseudo-gradient sequence unchanged; in a
nonconstant-input run, cadence may change the inputs and the conclusion need
not hold.
-/

namespace LeanMechanism

noncomputable section

/-- `cadence` is a partition of `budget` into positive inner-step blocks. -/
def IsInnerBudgetPartition (budget : Nat) (cadence : List Nat) : Prop :=
  cadence.sum = budget ∧ ∀ h ∈ cadence, 0 < h

/-- Number of outer updates induced by a cadence. -/
def cadenceOuterSteps (cadence : List Nat) : Nat := cadence.length

/-- Bias-corrected accumulated multiplier for a cadence in the constant-input
model.  By construction, only the number of outer updates is observable to this
model; the theorems below expose that factorization and its closed form. -/
def correctedCadenceMultiplier (cadence : List Nat) (mu : Real) : Real :=
  correctedEffectiveCoeff (cadenceOuterSteps cadence) mu

/-- Corrected frozen-gradient optimum associated with a cadence. -/
def correctedCadenceOptimalEta
    (cadence : List Nat) (mu a : Real) : Real :=
  correctedAlignedOptimalEta (cadenceOuterSteps cadence) mu a

/-- A positive total budget has no empty valid partition. -/
theorem partition_nonempty_of_budget_pos
    {budget : Nat} {cadence : List Nat}
    (hpart : IsInnerBudgetPartition budget cadence) (hbudget : 0 < budget) :
    cadence ≠ [] := by
  intro hempty
  subst cadence
  simp [IsInnerBudgetPartition] at hpart
  omega

/-- Consequently, a valid positive-budget cadence has at least one outer
step. -/
theorem cadenceOuterSteps_pos_of_partition
    {budget : Nat} {cadence : List Nat}
    (hpart : IsInnerBudgetPartition budget cadence) (hbudget : 0 < budget) :
    0 < cadenceOuterSteps cadence := by
  unfold cadenceOuterSteps
  exact List.length_pos_iff.mpr
    (partition_nonempty_of_budget_pos hpart hbudget)

/-- Closed form: the corrected accumulated multiplier is exactly
`T/(1-mu)`, with `T = cadence.length`; individual block sizes disappear. -/
theorem correctedCadenceMultiplier_closed_form
    (cadence : List Nat) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    correctedCadenceMultiplier cadence mu =
      (cadenceOuterSteps cadence : Real) / (1 - mu) := by
  unfold correctedCadenceMultiplier
  exact correctedEffectiveCoeff_closed_form
    (cadenceOuterSteps cadence) mu hmu0 hmu1

/-- Closed form for the corrected frozen-gradient optimum: it depends on a
cadence only through its outer-step count `T`. -/
theorem correctedCadenceOptimalEta_closed_form
    (cadence : List Nat) (mu a : Real)
    (hT : 0 < cadenceOuterSteps cadence)
    (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) (ha : 0 < a) :
    correctedCadenceOptimalEta cadence mu a =
      (1 - mu) / (a * (cadenceOuterSteps cadence : Real)) := by
  unfold correctedCadenceOptimalEta
  exact correctedAlignedOptimalEta_closed_form
    (cadenceOuterSteps cadence) mu a hT hmu0 hmu1 ha

/-- Strong factorization statement: even without assuming equal total inner
work, two cadence patterns with the same number of outer updates have identical
corrected accumulated multipliers in the constant-input model. -/
theorem correctedCadenceMultiplier_eq_of_same_outer_count
    (cadence₁ cadence₂ : List Nat) (mu : Real)
    (hcount : cadenceOuterSteps cadence₁ = cadenceOuterSteps cadence₂) :
    correctedCadenceMultiplier cadence₁ mu =
      correctedCadenceMultiplier cadence₂ mu := by
  unfold correctedCadenceMultiplier
  rw [hcount]

/-- The corresponding corrected optimum also factors only through the outer
step count. -/
theorem correctedCadenceOptimalEta_eq_of_same_outer_count
    (cadence₁ cadence₂ : List Nat) (mu a : Real)
    (hcount : cadenceOuterSteps cadence₁ = cadenceOuterSteps cadence₂) :
    correctedCadenceOptimalEta cadence₁ mu a =
      correctedCadenceOptimalEta cadence₂ mu a := by
  unfold correctedCadenceOptimalEta
  rw [hcount]

/-- **Elastic-cadence invariance at fixed budget and outer count.**

Any two positive re-partitions of the same inner-step budget into the same
number `T` of outer updates have equal corrected accumulated multipliers and
equal corrected frozen-gradient optima.  The budget hypotheses document the
elastic-run interpretation; algebraically, equality of `T` is the operative
condition.
-/
theorem elasticCadence_invariance
    (budget T : Nat) (cadence₁ cadence₂ : List Nat) (mu a : Real)
    (_hpart₁ : IsInnerBudgetPartition budget cadence₁)
    (_hpart₂ : IsInnerBudgetPartition budget cadence₂)
    (hcount₁ : cadenceOuterSteps cadence₁ = T)
    (hcount₂ : cadenceOuterSteps cadence₂ = T) :
    correctedCadenceMultiplier cadence₁ mu =
        correctedCadenceMultiplier cadence₂ mu ∧
      correctedCadenceOptimalEta cadence₁ mu a =
        correctedCadenceOptimalEta cadence₂ mu a := by
  have hcount : cadenceOuterSteps cadence₁ = cadenceOuterSteps cadence₂ :=
    hcount₁.trans hcount₂.symm
  exact ⟨
    correctedCadenceMultiplier_eq_of_same_outer_count
      cadence₁ cadence₂ mu hcount,
    correctedCadenceOptimalEta_eq_of_same_outer_count
      cadence₁ cadence₂ mu a hcount⟩

/-- Fixed-budget specialization with the optimum displayed explicitly.  This
is the paper-facing "depends on outer count, not cadence pattern" statement. -/
theorem elasticCadence_optimum_eq_count_formula
    (budget T : Nat) (cadence : List Nat) (mu a : Real)
    (hbudget : 0 < budget)
    (hpart : IsInnerBudgetPartition budget cadence)
    (hcount : cadenceOuterSteps cadence = T)
    (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) (ha : 0 < a) :
    correctedCadenceOptimalEta cadence mu a =
      (1 - mu) / (a * (T : Real)) := by
  have hTcadence : 0 < cadenceOuterSteps cadence :=
    cadenceOuterSteps_pos_of_partition hpart hbudget
  rw [correctedCadenceOptimalEta_closed_form
    cadence mu a hTcadence hmu0 hmu1 ha, hcount]

/-- The matching accumulated-multiplier formula for a fixed-budget cadence. -/
theorem elasticCadence_multiplier_eq_count_formula
    (budget T : Nat) (cadence : List Nat) (mu : Real)
    (_hpart : IsInnerBudgetPartition budget cadence)
    (hcount : cadenceOuterSteps cadence = T)
    (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    correctedCadenceMultiplier cadence mu = (T : Real) / (1 - mu) := by
  rw [correctedCadenceMultiplier_closed_form cadence mu hmu0 hmu1, hcount]

end

end LeanMechanism
