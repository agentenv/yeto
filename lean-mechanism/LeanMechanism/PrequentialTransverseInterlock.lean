import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Prequential transverse-interlock geometry

This file formalizes only the one-step Euclidean geometry used by the proposed
PTI-SGD mechanism.  It proves no convergence, finite-loss, or empirical
superiority claim.
-/

namespace LeanMechanism

noncomputable section

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The dimensionless normalization factor for a transverse coefficient. -/
def transverseScale (alpha : ℝ) : ℝ :=
  Real.sqrt (1 + alpha ^ 2)

/-- A norm-grafted turn by `alpha` in transverse direction `p` from `u`. -/
def transverseCandidate (u p : E) (alpha : ℝ) : E :=
  (1 / transverseScale alpha) • (u + alpha • p)

/-- Alignment gain of the transverse candidate against a subsequently revealed
direction `v`.  When `u`, the candidate, and `v` are unit vectors, this is the
corresponding cosine gain. -/
def transverseAlignmentGain (u p v : E) (alpha : ℝ) : ℝ :=
  ⟪transverseCandidate u p alpha, v⟫_ℝ - ⟪u, v⟫_ℝ

/-- The PTI normalization factor is positive for every real coefficient. -/
theorem transverseScale_pos (alpha : ℝ) : 0 < transverseScale alpha := by
  rw [transverseScale, Real.sqrt_pos]
  linarith [sq_nonneg alpha]

/-- The unnormalized sum of orthonormal baseline and transverse directions has
exactly the PTI normalization factor as its norm. -/
theorem norm_add_transverse_smul (u p : E) (alpha : ℝ)
    (hu : ‖u‖ = 1) (hp : ‖p‖ = 1) (hup : ⟪u, p⟫_ℝ = 0) :
    ‖u + alpha • p‖ = transverseScale alpha := by
  have hpu : ⟪p, u⟫_ℝ = 0 := by
    rw [real_inner_comm, hup]
  have hsquare : ‖u + alpha • p‖ ^ 2 = 1 + alpha ^ 2 := by
    rw [← real_inner_self_eq_norm_sq]
    simp only [inner_add_left, inner_add_right, real_inner_smul_left,
      real_inner_smul_right]
    rw [real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq, hu, hp, hup, hpu]
    ring
  calc
    ‖u + alpha • p‖ = Real.sqrt (‖u + alpha • p‖ ^ 2) :=
      (Real.sqrt_sq (norm_nonneg _)).symm
    _ = transverseScale alpha := by rw [hsquare]; rfl

/-- Under the orthonormal-turn hypotheses, `transverseCandidate` really is a
unit direction, so its inner product with a unit target is a cosine. -/
theorem norm_transverseCandidate (u p : E) (alpha : ℝ)
    (hu : ‖u‖ = 1) (hp : ‖p‖ = 1) (hup : ⟪u, p⟫_ℝ = 0) :
    ‖transverseCandidate u p alpha‖ = 1 := by
  rw [transverseCandidate, norm_smul, norm_add_transverse_smul u p alpha hu hp hup]
  have hs := transverseScale_pos alpha
  rw [Real.norm_eq_abs, abs_of_pos (one_div_pos.mpr hs), one_div,
    inv_mul_cancel₀ hs.ne']

/-- Exact normalized alignment-gain identity.  Orthogonality is not needed for
the algebraic identity itself; it is needed by `norm_transverseCandidate` to
interpret the left-hand side as cosine gain. -/
theorem transverseAlignmentGain_identity (u p v : E) (alpha : ℝ) :
    transverseAlignmentGain u p v alpha =
      (⟪u, v⟫_ℝ + alpha * ⟪p, v⟫_ℝ) / transverseScale alpha -
        ⟪u, v⟫_ℝ := by
  simp only [transverseAlignmentGain, transverseCandidate, real_inner_smul_left,
    inner_add_left]
  ring

/-- Because the normalization factor is positive, a transverse action improves
alignment exactly when its signed transverse contribution exceeds the radial
normalization penalty. -/
theorem transverseAlignmentGain_pos_iff (u p v : E) (alpha : ℝ) :
    0 < transverseAlignmentGain u p v alpha ↔
      alpha * ⟪p, v⟫_ℝ >
        ⟪u, v⟫_ℝ * (transverseScale alpha - 1) := by
  rw [transverseAlignmentGain_identity, sub_pos,
    lt_div_iff₀ (transverseScale_pos alpha)]
  constructor <;> intro h <;> linarith

/-- Stationary-next-direction counterexample: if the next direction is exactly
the unit baseline, every nonzero orthogonal transverse coefficient strictly
reduces alignment.  This rules out any unconditional "turning helps" theorem. -/
theorem transverseCandidate_stationary_alignment_lt (u p : E) (alpha : ℝ)
    (hu : ‖u‖ = 1) (hup : ⟪u, p⟫_ℝ = 0) (halpha : alpha ≠ 0) :
    ⟪transverseCandidate u p alpha, u⟫_ℝ < ⟪u, u⟫_ℝ := by
  have hpu : ⟪p, u⟫_ℝ = 0 := by
    rw [real_inner_comm, hup]
  have huu : ⟪u, u⟫_ℝ = 1 := by
    rw [real_inner_self_eq_norm_sq, hu]
    norm_num
  have halpha_sq : 0 < alpha ^ 2 := sq_pos_of_ne_zero halpha
  have hscale : 1 < transverseScale alpha := by
    have harg : (1 : ℝ) < 1 + alpha ^ 2 := by linarith
    calc
      1 = Real.sqrt 1 := by norm_num
      _ < Real.sqrt (1 + alpha ^ 2) :=
        Real.sqrt_lt_sqrt (by norm_num) harg
      _ = transverseScale alpha := rfl
  calc
    ⟪transverseCandidate u p alpha, u⟫_ℝ = 1 / transverseScale alpha := by
      simp [transverseCandidate, real_inner_smul_left, inner_add_left, hu, hpu]
    _ < 1 := by
      simpa using one_div_lt_one_div_of_lt zero_lt_one hscale
    _ = ⟪u, u⟫_ℝ := huu.symm

/-- The same stationary counterexample stated directly as a negative alignment
gain. -/
theorem transverseAlignmentGain_stationary_neg (u p : E) (alpha : ℝ)
    (hu : ‖u‖ = 1) (hup : ⟪u, p⟫_ℝ = 0) (halpha : alpha ≠ 0) :
    transverseAlignmentGain u p u alpha < 0 := by
  rw [transverseAlignmentGain, sub_neg]
  exact transverseCandidate_stationary_alignment_lt u p alpha hu hup halpha

/-- With a unit transverse direction, the stationary example is a literal
cosine counterexample: the candidate stays unit-norm but is less aligned with
the unchanged unit target. -/
theorem transverseCandidate_stationary_cosine_counterexample
    (u p : E) (alpha : ℝ) (hu : ‖u‖ = 1) (hp : ‖p‖ = 1)
    (hup : ⟪u, p⟫_ℝ = 0) (halpha : alpha ≠ 0) :
    ‖transverseCandidate u p alpha‖ = 1 ∧
      ⟪transverseCandidate u p alpha, u⟫_ℝ < ⟪u, u⟫_ℝ := by
  exact ⟨norm_transverseCandidate u p alpha hu hp hup,
    transverseCandidate_stationary_alignment_lt u p alpha hu hup halpha⟩

end

end LeanMechanism
