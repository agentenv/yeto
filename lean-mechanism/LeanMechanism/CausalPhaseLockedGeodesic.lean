import LeanMechanism.CausalGeodesicContinuation
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Tactic.Module

/-!
# Causal phase-locked geodesic geometry

This file formalizes the angle-based CPLG proposal exactly.  Its commanded
angle is

`phi = max 0 coherence * min thetaNow thetaPrevious (arctan (1 / 4))`,

and its candidate is the great-circle point

`cos phi • stock + sin phi • continuationTangent`.

The distinction from normalized-linear CGC is essential: passing an angle
`phi` as CGC's tangent ratio would realize angle `arctan phi`, not `phi`.

The statements below prove only deterministic geometry: the exact sphere
parallel-transport formula used by the coherence score, stock degeneration,
unit norm, the angle cap, ideal constant-rotation recovery, and a reversal
counterexample.  They prove no stochastic convergence, finite-loss
improvement, or superiority over SGD.
-/

namespace LeanMechanism

noncomputable section

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Parallel transport of a tangent vector from `previous` to `current` along
the short great-circle arc on the unit sphere.  The antipodal case, where the
denominator vanishes, is deliberately outside the CPLG admission contract. -/
def sphereParallelTransport (previous current tangent : E) : E :=
  tangent -
    (⟪tangent, current⟫_ℝ / (1 + ⟪previous, current⟫_ℝ)) •
      (previous + current)

/-- Phase coherence compares the current continuation tangent with the
previous continuation tangent after exact sphere parallel transport. -/
def phaseCoherence
    (previous current previousTangent currentTangent : E) : ℝ :=
  ⟪currentTangent,
    sphereParallelTransport previous current previousTangent⟫_ℝ

/-- Only positive agreement between consecutive transported tangents may
contribute to the CPLG angle. -/
def phaseLock (coherence : ℝ) : ℝ :=
  max 0 coherence

/-- The exact CPLG command angle.  Inputs `thetaNow` and `thetaPrevious` are
angles in radians, not tangent ratios. -/
def cplgAngle (thetaNow thetaPrevious coherence : ℝ) : ℝ :=
  phaseLock coherence *
    min (min thetaNow thetaPrevious) (Real.arctan (1 / 4))

/-- The exact angle-based phase-locked great-circle candidate. -/
def cplgCandidate (stock continuationTangent : E)
    (thetaNow thetaPrevious coherence : ℝ) : E :=
  let phi := cplgAngle thetaNow thetaPrevious coherence
  Real.cos phi • stock + Real.sin phi • continuationTangent

theorem phaseLock_nonneg (coherence : ℝ) : 0 ≤ phaseLock coherence := by
  exact le_max_left 0 coherence

theorem phaseLock_eq_zero_of_nonpos (coherence : ℝ) (hcoherence : coherence ≤ 0) :
    phaseLock coherence = 0 := by
  simp [phaseLock, hcoherence]

/-- The transport formula lands in the tangent space at `current`. -/
theorem inner_sphereParallelTransport_current
    (previous current tangent : E)
    (hcurrent : ‖current‖ = 1)
    (hdenominator : 1 + ⟪previous, current⟫_ℝ ≠ 0) :
    ⟪sphereParallelTransport previous current tangent, current⟫_ℝ = 0 := by
  have hcurrentSelf : ⟪current, current⟫_ℝ = 1 := by
    rw [real_inner_self_eq_norm_sq, hcurrent]
    norm_num
  unfold sphereParallelTransport
  rw [inner_sub_left, real_inner_smul_left, inner_add_left, hcurrentSelf]
  field_simp
  ring

/-- In a two-plane constant rotation, exact sphere transport carries the
previous forward tangent to the current forward tangent.  Here `backward` is
the current backward tangent, so the transported forward tangent is
`-backward`. -/
theorem sphereParallelTransport_constant_rotation
    (current backward : E) (rho s : ℝ)
    (hcurrent : ‖current‖ = 1)
    (horthogonal : ⟪current, backward⟫_ℝ = 0)
    (hcircle : rho ^ 2 + s ^ 2 = 1)
    (hdenominator : 1 + rho ≠ 0) :
    sphereParallelTransport
        (rho • current + s • backward)
        current
        (s • current - rho • backward) =
      -backward := by
  have hbackwardCurrent : ⟪backward, current⟫_ℝ = 0 := by
    rw [real_inner_comm, horthogonal]
  have hpreviousCurrent :
      ⟪rho • current + s • backward, current⟫_ℝ = rho := by
    rw [inner_add_left, real_inner_smul_left, real_inner_smul_left,
      real_inner_self_eq_norm_sq, hcurrent, hbackwardCurrent]
    ring
  have htangentCurrent :
      ⟪s • current - rho • backward, current⟫_ℝ = s := by
    rw [inner_sub_left, real_inner_smul_left, real_inner_smul_left,
      real_inner_self_eq_norm_sq, hcurrent, hbackwardCurrent]
    ring
  let k : ℝ := s / (1 + rho)
  have huCoefficient : s - k * (rho + 1) = 0 := by
    dsimp [k]
    field_simp
    ring
  have hbCoefficient : -rho - k * s = -1 := by
    dsimp [k]
    field_simp
    nlinarith
  rw [sphereParallelTransport, hpreviousCurrent, htangentCurrent]
  change
    (s • current - rho • backward) -
        k • ((rho • current + s • backward) + current) =
      -backward
  calc
    (s • current - rho • backward) -
        k • ((rho • current + s • backward) + current) =
        (s - k * (rho + 1)) • current +
          (-rho - k * s) • backward := by module
    _ = -backward := by rw [huCoefficient, hbCoefficient]; simp

/-- Consequently the exact transported-tangent coherence is one in the ideal
constant-rotation model. -/
theorem phaseCoherence_constant_rotation
    (current backward : E) (rho s : ℝ)
    (hcurrent : ‖current‖ = 1) (hbackward : ‖backward‖ = 1)
    (horthogonal : ⟪current, backward⟫_ℝ = 0)
    (hcircle : rho ^ 2 + s ^ 2 = 1)
    (hdenominator : 1 + rho ≠ 0) :
    phaseCoherence
        (rho • current + s • backward)
        current
        (s • current - rho • backward)
        (-backward) = 1 := by
  rw [phaseCoherence,
    sphereParallelTransport_constant_rotation current backward rho s
      hcurrent horthogonal hcircle hdenominator,
    real_inner_self_eq_norm_sq, norm_neg, hbackward]
  norm_num

theorem cplgAngle_nonneg (thetaNow thetaPrevious coherence : ℝ)
    (hthetaNow : 0 ≤ thetaNow) (hthetaPrevious : 0 ≤ thetaPrevious) :
    0 ≤ cplgAngle thetaNow thetaPrevious coherence := by
  have hcap : 0 ≤ Real.arctan (1 / 4 : ℝ) :=
    (Real.arctan_pos.mpr (by norm_num)).le
  have hinner : 0 ≤ min thetaNow thetaPrevious :=
    le_min hthetaNow hthetaPrevious
  exact mul_nonneg (phaseLock_nonneg coherence) (le_min hinner hcap)

/-- For coherence in `[0,1]`, the actual commanded angle—not a proxy ratio—is
bounded by `arctan (1/4)`. -/
theorem cplgAngle_le_cap (thetaNow thetaPrevious coherence : ℝ)
    (hthetaNow : 0 ≤ thetaNow) (hthetaPrevious : 0 ≤ thetaPrevious)
    (hcoherence0 : 0 ≤ coherence) (hcoherence1 : coherence ≤ 1) :
    cplgAngle thetaNow thetaPrevious coherence ≤ Real.arctan (1 / 4) := by
  have hcap : 0 ≤ Real.arctan (1 / 4 : ℝ) :=
    (Real.arctan_pos.mpr (by norm_num)).le
  have hinner : 0 ≤ min thetaNow thetaPrevious :=
    le_min hthetaNow hthetaPrevious
  have hturn0 :
      0 ≤ min (min thetaNow thetaPrevious) (Real.arctan (1 / 4)) :=
    le_min hinner hcap
  have hturn1 :
      min (min thetaNow thetaPrevious) (Real.arctan (1 / 4)) ≤
        Real.arctan (1 / 4) :=
    min_le_right _ _
  have hslack :
      0 ≤ (1 - coherence) *
        min (min thetaNow thetaPrevious) (Real.arctan (1 / 4)) :=
    mul_nonneg (sub_nonneg.mpr hcoherence1) hturn0
  rw [cplgAngle, phaseLock, max_eq_right hcoherence0]
  nlinarith

/-- Nonpositive transported-tangent coherence gives exact stock. -/
theorem cplgCandidate_stock_of_nonpositive_coherence
    (stock continuationTangent : E)
    (thetaNow thetaPrevious coherence : ℝ)
    (hcoherence : coherence ≤ 0) :
    cplgCandidate stock continuationTangent thetaNow thetaPrevious coherence = stock := by
  have hphi : cplgAngle thetaNow thetaPrevious coherence = 0 := by
    rw [cplgAngle, phaseLock_eq_zero_of_nonpos coherence hcoherence, zero_mul]
  simp [cplgCandidate, hphi]

/-- A zero current angle gives exact stock. -/
theorem cplgCandidate_stock_of_zero_current
    (stock continuationTangent : E) (thetaPrevious coherence : ℝ)
    (hthetaPrevious : 0 ≤ thetaPrevious) :
    cplgCandidate stock continuationTangent 0 thetaPrevious coherence = stock := by
  have hturn :
      min (min (0 : ℝ) thetaPrevious) (Real.arctan (1 / 4)) = 0 := by
    rw [min_eq_left hthetaPrevious,
      min_eq_left (Real.arctan_nonneg.mpr (by norm_num))]
  have hphi : cplgAngle 0 thetaPrevious coherence = 0 := by
    rw [cplgAngle, hturn, mul_zero]
  simp [cplgCandidate, hphi]

/-- A zero preceding angle also gives exact stock. -/
theorem cplgCandidate_stock_of_zero_previous
    (stock continuationTangent : E) (thetaNow coherence : ℝ)
    (hthetaNow : 0 ≤ thetaNow) :
    cplgCandidate stock continuationTangent thetaNow 0 coherence = stock := by
  have hturn :
      min (min thetaNow (0 : ℝ)) (Real.arctan (1 / 4)) = 0 := by
    rw [min_eq_right hthetaNow,
      min_eq_left (Real.arctan_nonneg.mpr (by norm_num))]
  have hphi : cplgAngle thetaNow 0 coherence = 0 := by
    rw [cplgAngle, hturn, mul_zero]
  simp [cplgCandidate, hphi]

/-- The exact angle-based great-circle candidate has unit norm under the
orthonormal stock/tangent hypotheses. -/
theorem norm_cplgCandidate (stock continuationTangent : E)
    (thetaNow thetaPrevious coherence : ℝ)
    (hstock : ‖stock‖ = 1) (htangent : ‖continuationTangent‖ = 1)
    (horthogonal : ⟪stock, continuationTangent⟫_ℝ = 0) :
    ‖cplgCandidate stock continuationTangent thetaNow thetaPrevious coherence‖ = 1 := by
  let phi := cplgAngle thetaNow thetaPrevious coherence
  have htangentStock : ⟪continuationTangent, stock⟫_ℝ = 0 := by
    rw [real_inner_comm, horthogonal]
  have hsquare :
      ‖Real.cos phi • stock + Real.sin phi • continuationTangent‖ ^ 2 = 1 := by
    rw [← real_inner_self_eq_norm_sq]
    simp only [inner_add_left, inner_add_right, real_inner_smul_left,
      real_inner_smul_right]
    rw [real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq,
      hstock, htangent, horthogonal, htangentStock]
    nlinarith [Real.cos_sq_add_sin_sq phi]
  rw [cplgCandidate]
  change ‖Real.cos phi • stock + Real.sin phi • continuationTangent‖ = 1
  calc
    ‖Real.cos phi • stock + Real.sin phi • continuationTangent‖ =
        Real.sqrt
          (‖Real.cos phi • stock + Real.sin phi • continuationTangent‖ ^ 2) :=
      (Real.sqrt_sq (norm_nonneg _)).symm
    _ = 1 := by rw [hsquare]; norm_num

/-- The stock-direction inner product is exactly the cosine of the commanded
angle, making `cplgAngle_le_cap` an actual angular safety statement. -/
theorem inner_cplgCandidate_stock (stock continuationTangent : E)
    (thetaNow thetaPrevious coherence : ℝ)
    (hstock : ‖stock‖ = 1)
    (horthogonal : ⟪stock, continuationTangent⟫_ℝ = 0) :
    ⟪cplgCandidate stock continuationTangent thetaNow thetaPrevious coherence,
        stock⟫_ℝ =
      Real.cos (cplgAngle thetaNow thetaPrevious coherence) := by
  have htangentStock : ⟪continuationTangent, stock⟫_ℝ = 0 := by
    rw [real_inner_comm, horthogonal]
  simp [cplgCandidate, inner_add_left, real_inner_smul_left, hstock, htangentStock]

/-- Under ideal constant angular motion, coherence one, and an inactive cap,
CPLG recovers the exact next great-circle direction. -/
theorem cplgCandidate_exact_constant_rotation
    (stock continuationTangent : E) (theta : ℝ)
    (htheta0 : 0 ≤ theta)
    (hcap : theta ≤ Real.arctan (1 / 4)) :
    cplgCandidate stock continuationTangent theta theta 1 =
        Real.cos theta • stock + Real.sin theta • continuationTangent ∧
      cplgAngle theta theta 1 = theta ∧
      0 ≤ cplgAngle theta theta 1 := by
  have hphi : cplgAngle theta theta 1 = theta := by
    have hlock : phaseLock 1 = 1 := by norm_num [phaseLock]
    rw [cplgAngle, hlock, min_self, min_eq_left hcap, one_mul]
  constructor
  · simp [cplgCandidate, hphi]
  · refine ⟨hphi, ?_⟩
    calc
      0 ≤ theta := htheta0
      _ = cplgAngle theta theta 1 := hphi.symm

/-- The transport and candidate results compose: in the ideal constant-rotation
plane, CPLG may consume the coherence computed by `phaseCoherence` itself and
recovers the next direction rather than relying on a caller-supplied `1`. -/
theorem cplgCandidate_exact_transported_constant_rotation
    (current backward : E) (rho s theta : ℝ)
    (hcurrent : ‖current‖ = 1) (hbackward : ‖backward‖ = 1)
    (horthogonal : ⟪current, backward⟫_ℝ = 0)
    (hcircle : rho ^ 2 + s ^ 2 = 1)
    (hdenominator : 1 + rho ≠ 0)
    (htheta0 : 0 ≤ theta)
    (hcap : theta ≤ Real.arctan (1 / 4)) :
    cplgCandidate current (-backward) theta theta
        (phaseCoherence
          (rho • current + s • backward)
          current
          (s • current - rho • backward)
          (-backward)) =
      Real.cos theta • current - Real.sin theta • backward := by
  rw [phaseCoherence_constant_rotation current backward rho s
    hcurrent hbackward horthogonal hcircle hdenominator]
  have hexact :=
    (cplgCandidate_exact_constant_rotation current (-backward) theta
      htheta0 hcap).1
  simpa [smul_neg, sub_eq_add_neg] using hexact

/-- A positive emitted turn is strictly worse than stock when the next
direction reverses exactly against the continuation tangent.  Phase coherence
is causal evidence, not a geometric guarantee about the next observation. -/
theorem cplg_reversal_alignment_lt
    (stock continuationTangent : E)
    (thetaNow thetaPrevious coherence : ℝ)
    (hstockTangent : ⟪stock, continuationTangent⟫_ℝ = 0)
    (htangent : ‖continuationTangent‖ = 1)
    (hthetaNow : 0 < thetaNow) (hthetaPrevious : 0 < thetaPrevious)
    (hcoherence0 : 0 < coherence) (hcoherence1 : coherence ≤ 1) :
    ⟪cplgCandidate stock continuationTangent thetaNow thetaPrevious coherence,
        -continuationTangent⟫_ℝ <
      ⟪stock, -continuationTangent⟫_ℝ := by
  have hcapPos : 0 < Real.arctan (1 / 4 : ℝ) :=
    Real.arctan_pos.mpr (by norm_num)
  have hturnPos :
      0 < min (min thetaNow thetaPrevious) (Real.arctan (1 / 4)) :=
    lt_min (lt_min hthetaNow hthetaPrevious) hcapPos
  have hphase : phaseLock coherence = coherence := max_eq_right hcoherence0.le
  have hphiPos : 0 < cplgAngle thetaNow thetaPrevious coherence := by
    rw [cplgAngle, hphase]
    exact mul_pos hcoherence0 hturnPos
  have hphiCap := cplgAngle_le_cap thetaNow thetaPrevious coherence
    hthetaNow.le hthetaPrevious.le hcoherence0.le hcoherence1
  have hphiPi : cplgAngle thetaNow thetaPrevious coherence < Real.pi :=
    hphiCap.trans_lt
      ((Real.arctan_lt_pi_div_two (1 / 4)).trans
        (half_lt_self Real.pi_pos))
  have hsin : 0 < Real.sin (cplgAngle thetaNow thetaPrevious coherence) :=
    Real.sin_pos_of_pos_of_lt_pi hphiPos hphiPi
  have hcandidateTangent :
      ⟪cplgCandidate stock continuationTangent thetaNow thetaPrevious coherence,
          continuationTangent⟫_ℝ =
        Real.sin (cplgAngle thetaNow thetaPrevious coherence) := by
    simp only [cplgCandidate, inner_add_left, real_inner_smul_left]
    rw [hstockTangent, real_inner_self_eq_norm_sq, htangent]
    norm_num
  rw [inner_neg_right, inner_neg_right, hcandidateTangent, hstockTangent, neg_zero]
  exact neg_lt_zero.mpr hsin

/-- Concrete non-equivalence certificate for the superseded ratio proxy:
feeding the positive CPLG cap angle directly into normalized-linear CGC would
realize `arctan cap`, which is strictly smaller than `cap`. -/
theorem cplg_angle_is_not_cgc_ratio_at_cap :
    Real.arctan (Real.arctan (1 / 4)) < Real.arctan (1 / 4) := by
  have hcapPos : 0 < Real.arctan (1 / 4 : ℝ) :=
    Real.arctan_pos.mpr (by norm_num)
  have hcapLtPiTwo : Real.arctan (1 / 4 : ℝ) < Real.pi / 2 :=
    Real.arctan_lt_pi_div_two _
  have hcapLtQuarter : Real.arctan (1 / 4 : ℝ) < 1 / 4 := by
    simpa using Real.lt_tan hcapPos hcapLtPiTwo
  exact Real.arctan_strictMono hcapLtQuarter

end

end LeanMechanism
