import Mathlib
import LeanMechanism.QuadraticAlignment

/-!
# Constant-momentum specialization of NAdam's product correction

Dozat Algorithm 8 corrects its first-moment lookahead term using a scheduled
momentum product of the form

`1 / (1 - product_(i=1)^(t+1) mu_i)`.

We represent that one-indexed product by iterating `i` over the zero-indexed
finite range `0, ..., t` and evaluating the schedule at `i+1`.  When every
`mu_i` is the same `mu`, the product is exactly `mu^(t+1)`, so the correction
on the lookahead term is exactly this project's `biasCorrectionScale t mu`.

The scope is intentionally limited to this scalar first-moment correction.
NAdam's full update also contains its current-gradient injection, an adaptive
second-moment denominator, epsilon handling, and possibly a nonconstant
momentum schedule.  No full-optimizer or trajectory equivalence is claimed.
-/

namespace LeanMechanism

noncomputable section

open Finset
open scoped BigOperators

/-- Dozat's one-indexed scheduled momentum product
`product_(i=1)^(t+1) mu_i`. -/
def nadamMuProduct (muSchedule : Nat → Real) (t : Nat) : Real :=
  ∏ i ∈ range (t + 1), muSchedule (i + 1)

/-- The (scalar) NAdam lookahead numerator `mu_(t+1) * m_t`. -/
def nadamLookaheadTerm
    (muSchedule : Nat → Real) (t : Nat) (firstMoment : Real) : Real :=
  muSchedule (t + 1) * firstMoment

/-- The product-bias-corrected NAdam lookahead contribution. -/
def nadamCorrectedLookahead
    (muSchedule : Nat → Real) (t : Nat) (firstMoment : Real) : Real :=
  nadamLookaheadTerm muSchedule t firstMoment /
    (1 - nadamMuProduct muSchedule t)

/-- A constant momentum schedule collapses Dozat's product to a power with
exactly `t+1` factors. -/
@[simp]
theorem nadamMuProduct_constant (t : Nat) (mu : Real) :
    nadamMuProduct (fun _i => mu) t = mu ^ (t + 1) := by
  simp [nadamMuProduct]

/-- **NAdam specialization.**  With `mu_i = mu`, Dozat Algorithm 8's product
correction on the lookahead term is exactly the outer correction factor
`1/(1-mu^(t+1))` applied to that same term.

The identity is algebraic and remains true even when the displayed denominator
is zero (Lean's field inverse is total); the intended optimizer regime is, as
elsewhere, `0 ≤ mu < 1`.
-/
theorem nadam_constant_mu_lookahead_eq_outer_correction
    (t : Nat) (mu firstMoment : Real) :
    nadamCorrectedLookahead (fun _i => mu) t firstMoment =
      biasCorrectionScale t mu *
        nadamLookaheadTerm (fun _i => mu) t firstMoment := by
  simp only [nadamCorrectedLookahead, nadamMuProduct_constant,
    biasCorrectionScale]
  unfold nadamLookaheadTerm
  ring

end

end LeanMechanism
