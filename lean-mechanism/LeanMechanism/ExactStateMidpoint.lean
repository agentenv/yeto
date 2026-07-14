import Mathlib

/-!
Narrow algebra used by the exact-state midpoint candidates. These statements
do not imply lower neural-network loss or stochastic convergence.
-/

namespace ExactStateMidpoint

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Richardson extrapolation is a fixed point on a stationary force. -/
theorem richardson_stationary (x : V) : (2 : ℝ) • x - x = x := by
  module

/-- A trust interpolation with coefficient in `[0,1]` cannot move farther
from the baseline than the proposed correction. -/
theorem trust_segment_bound (g δ : V) (α : ℝ)
    (hα0 : 0 ≤ α) (hα1 : α ≤ 1) :
    ‖(g + α • δ) - g‖ ≤ ‖δ‖ := by
  simp only [add_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_nonneg hα0]
  exact mul_le_of_le_one_left (norm_nonneg δ) hα1

/-- A stationary observed half-path gives a zero MSTP trust radius. -/
theorem stationary_turn_radius (A B : V) (h : A = B) :
    min ‖A + B‖ ‖B - A‖ = 0 := by
  subst B
  simp

end ExactStateMidpoint
