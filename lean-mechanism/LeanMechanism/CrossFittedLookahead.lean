import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Cross-fitted lookahead geometry

This file formalizes one scalar alignment fact used by the proposed CFLX-SGD
mechanism and one counterexample limiting its interpretation.  It proves no
finite-loss, population, or convergence result.
-/

namespace LeanMechanism

/-- Suppose the observed unit lookahead direction decomposes in an orthonormal
stock/transverse plane with positive stock coefficient `a` and transverse
coefficient `b`.  If a positive turn `beta` does not rotate past that direction
(`beta * a ≤ b`), normalization still leaves the candidate strictly better
aligned with the observed lookahead direction than the stock direction.

This is only the scalar inequality underlying the geometric statement. -/
theorem clippedLookahead_improves_alignment (a b beta : ℝ)
    (ha : 0 < a) (hbeta : 0 < beta) (hclip : beta * a ≤ b) :
    a < (a + beta * b) / Real.sqrt (1 + beta ^ 2) := by
  have hbeta_sq : 0 < beta ^ 2 := sq_pos_of_pos hbeta
  have harg : 1 < 1 + beta ^ 2 := by linarith
  have harg_pos : 0 < 1 + beta ^ 2 := by linarith
  have hsqrt_pos : 0 < Real.sqrt (1 + beta ^ 2) :=
    Real.sqrt_pos.2 harg_pos
  have hsqrt_lt : Real.sqrt (1 + beta ^ 2) < 1 + beta ^ 2 := by
    rw [Real.sqrt_lt' harg_pos]
    nlinarith
  have htransverse : a * beta ^ 2 ≤ beta * b := by
    have hscaled := mul_le_mul_of_nonneg_left hclip hbeta.le
    nlinarith
  apply (lt_div_iff₀ hsqrt_pos).2
  calc
    a * Real.sqrt (1 + beta ^ 2) < a * (1 + beta ^ 2) :=
      mul_lt_mul_of_pos_left hsqrt_lt ha
    _ = a + a * beta ^ 2 := by ring
    _ ≤ a + beta * b := by linarith

/-- Auxiliary proposal loss used by the concrete distribution-shift
counterexample. -/
def lookaheadProbeLoss (x : ℝ) : ℝ :=
  (x - 1) ^ 2

/-- True/evaluation loss used by the concrete distribution-shift
counterexample. -/
def lookaheadTrueLoss (x : ℝ) : ℝ :=
  (x + 1) ^ 2

/-- A finite probe certificate does not imply improvement on a different true
or evaluation objective: the probe strictly prefers candidate `1` to stock
`0`, while the true loss strictly prefers stock `0` to candidate `1`.

This counterexample is why CFLX-SGD still needs disjoint audit data, exact CRN
finite-loss replay, and fresh matched SGD controls. -/
theorem lookaheadProbe_does_not_imply_true_improvement :
    lookaheadProbeLoss 1 < lookaheadProbeLoss 0 ∧
      lookaheadTrueLoss 0 < lookaheadTrueLoss 1 := by
  norm_num [lookaheadProbeLoss, lookaheadTrueLoss]

end LeanMechanism
