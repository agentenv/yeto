import LeanMechanism.CausalGeodesicContinuation

/-!
# Causal phase-locked geodesic geometry

This file formalizes a conservative extension of causal geodesic continuation.
The mechanism receives two nonnegative observed tangent ratios and a causal
coherence score.  It turns only by the smaller observed ratio, attenuated by
the nonnegative part of coherence and capped at `1/4`.

The statements below are exact Euclidean geometry.  They prove stock
degeneration, unit-norm grafting, the fixed angular cap, and exact recovery of
an ideal constant rotation below the cap.  The final theorem records an honest
reversal counterexample.  Nothing here proves stochastic convergence,
finite-loss improvement, or superiority over SGD.
-/

namespace LeanMechanism

noncomputable section

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Only positive agreement between consecutive transported tangents is
allowed to contribute to the turn. -/
def phaseLock (coherence : ℝ) : ℝ :=
  max 0 coherence

/-- The phase-locked tangent ratio.  `currentTurn` and `previousTurn` are
dimensionless nonnegative tangent ratios; `coherence` is intended to lie in
`[-1, 1]`. -/
def cplgRatio (currentTurn previousTurn coherence : ℝ) : ℝ :=
  phaseLock coherence * min (min currentTurn previousTurn) (1 / 4)

/-- CPLG applies the phase-locked ratio through the same norm-grafted
continuation kernel as CGC. -/
def cplgCandidate (u backwardTangent : E)
    (currentTurn previousTurn coherence : ℝ) : E :=
  cgcCandidate u backwardTangent (cplgRatio currentTurn previousTurn coherence)

theorem phaseLock_nonneg (coherence : ℝ) : 0 ≤ phaseLock coherence := by
  exact le_max_left 0 coherence

theorem phaseLock_eq_zero_of_nonpos (coherence : ℝ) (hcoherence : coherence ≤ 0) :
    phaseLock coherence = 0 := by
  simp [phaseLock, hcoherence]

/-- Nonnegative observed turns always produce a nonnegative continuation
ratio; CPLG never silently reverses the declared continuation tangent. -/
theorem cplgRatio_nonneg (currentTurn previousTurn coherence : ℝ)
    (hcurrent : 0 ≤ currentTurn) (hprevious : 0 ≤ previousTurn) :
    0 ≤ cplgRatio currentTurn previousTurn coherence := by
  have hinner : 0 ≤ min currentTurn previousTurn := le_min hcurrent hprevious
  have hturn : 0 ≤ min (min currentTurn previousTurn) (1 / 4 : ℝ) :=
    le_min hinner (by norm_num)
  exact mul_nonneg (phaseLock_nonneg coherence) hturn

/-- With coherence in `[0, 1]`, the phase-locked ratio remains below the
campaign-wide `1/4` safety cap. -/
theorem cplgRatio_le_quarter (currentTurn previousTurn coherence : ℝ)
    (hcurrent : 0 ≤ currentTurn) (hprevious : 0 ≤ previousTurn)
    (hcoherence0 : 0 ≤ coherence) (hcoherence1 : coherence ≤ 1) :
    cplgRatio currentTurn previousTurn coherence ≤ (1 / 4 : ℝ) := by
  have hinner : 0 ≤ min currentTurn previousTurn := le_min hcurrent hprevious
  have hturn0 : 0 ≤ min (min currentTurn previousTurn) (1 / 4 : ℝ) :=
    le_min hinner (by norm_num)
  have hturn1 : min (min currentTurn previousTurn) (1 / 4 : ℝ) ≤ 1 / 4 :=
    min_le_right _ _
  have hslack :
      0 ≤ (1 - coherence) * min (min currentTurn previousTurn) (1 / 4 : ℝ) :=
    mul_nonneg (sub_nonneg.mpr hcoherence1) hturn0
  rw [cplgRatio, phaseLock, max_eq_right hcoherence0]
  nlinarith

/-- Nonpositive transported-tangent coherence disables the turn exactly. -/
theorem cplgCandidate_stock_of_nonpositive_coherence
    (u backwardTangent : E) (currentTurn previousTurn coherence : ℝ)
    (hcoherence : coherence ≤ 0) :
    cplgCandidate u backwardTangent currentTurn previousTurn coherence = u := by
  rw [cplgCandidate, cplgRatio,
    phaseLock_eq_zero_of_nonpos coherence hcoherence, zero_mul,
    cgcCandidate_zero]

/-- A zero current angular observation disables the turn exactly. -/
theorem cplgCandidate_stock_of_zero_current
    (u backwardTangent : E) (previousTurn coherence : ℝ)
    (hprevious : 0 ≤ previousTurn) :
    cplgCandidate u backwardTangent 0 previousTurn coherence = u := by
  have hturn : min (min (0 : ℝ) previousTurn) (1 / 4) = 0 := by
    rw [min_eq_left hprevious]
    norm_num
  rw [cplgCandidate, cplgRatio, hturn, mul_zero, cgcCandidate_zero]

/-- A zero preceding angular observation also disables the turn exactly. -/
theorem cplgCandidate_stock_of_zero_previous
    (u backwardTangent : E) (currentTurn coherence : ℝ)
    (hcurrent : 0 ≤ currentTurn) :
    cplgCandidate u backwardTangent currentTurn 0 coherence = u := by
  have hturn : min (min currentTurn (0 : ℝ)) (1 / 4) = 0 := by
    rw [min_eq_right hcurrent]
    norm_num
  rw [cplgCandidate, cplgRatio, hturn, mul_zero, cgcCandidate_zero]

/-- Phase locking does not disturb CGC's exact unit-norm graft. -/
theorem norm_cplgCandidate (u backwardTangent : E)
    (currentTurn previousTurn coherence : ℝ)
    (hu : ‖u‖ = 1) (hb : ‖backwardTangent‖ = 1)
    (hub : ⟪u, backwardTangent⟫_ℝ = 0) :
    ‖cplgCandidate u backwardTangent currentTurn previousTurn coherence‖ = 1 := by
  exact norm_cgcCandidate u backwardTangent
    (cplgRatio currentTurn previousTurn coherence) hu hb hub

/-- Under the declared nonnegative-turn and `[0,1]` coherence contract, CPLG
inherits the squared stock-cosine lower bound `16/17`. -/
theorem cplgCandidate_cap_cosine_sq (u backwardTangent : E)
    (currentTurn previousTurn coherence : ℝ)
    (hu : ‖u‖ = 1) (hub : ⟪u, backwardTangent⟫_ℝ = 0)
    (hcurrent : 0 ≤ currentTurn) (hprevious : 0 ≤ previousTurn)
    (hcoherence0 : 0 ≤ coherence) (hcoherence1 : coherence ≤ 1) :
    (16 / 17 : ℝ) ≤
      ⟪cplgCandidate u backwardTangent currentTurn previousTurn coherence, u⟫_ℝ ^ 2 := by
  have hratio0 := cplgRatio_nonneg currentTurn previousTurn coherence hcurrent hprevious
  have hratio1 := cplgRatio_le_quarter currentTurn previousTurn coherence
    hcurrent hprevious hcoherence0 hcoherence1
  exact cgcCandidate_cap_cosine_sq u backwardTangent
    (cplgRatio currentTurn previousTurn coherence) hu hub (by linarith) hratio1

/-- In the ideal constant-rotation representation, both observed tangent
ratios are `s/rho`, coherence is one, and the cap is inactive.  CPLG then
recovers the exact next unit direction `rho • u - s • backwardTangent`. -/
theorem cplgCandidate_exact_constant_rotation
    (u backwardTangent : E) (rho s : ℝ)
    (hrho : 0 < rho) (hs : 0 ≤ s)
    (hunit : rho ^ 2 + s ^ 2 = 1)
    (hcap : s / rho ≤ (1 / 4 : ℝ)) :
    cplgCandidate u backwardTangent (s / rho) (s / rho) 1 =
      rho • u - s • backwardTangent ∧
      0 ≤ cplgRatio (s / rho) (s / rho) 1 := by
  have hturn0 : 0 ≤ s / rho := div_nonneg hs hrho.le
  constructor
  · have hratio : cplgRatio (s / rho) (s / rho) 1 = s / rho := by
      have hlock : phaseLock 1 = 1 := by norm_num [phaseLock]
      rw [cplgRatio, hlock, min_self, min_eq_left hcap, one_mul]
    rw [cplgCandidate, hratio]
    exact cgcCandidate_exact_continuation u backwardTangent rho s hrho hunit
  · exact cplgRatio_nonneg (s / rho) (s / rho) 1 hturn0 hturn0

/-- Honest non-dominance statement: whenever CPLG emits a positive continuation
turn but the next direction instead reverses toward the old backward tangent,
its alignment is strictly worse than stock.  Causal phase evidence can guard
this case empirically; geometry alone cannot rule it out. -/
theorem cplg_reversal_alignment_lt
    (u backwardTangent : E) (currentTurn previousTurn coherence : ℝ)
    (hub : ⟪u, backwardTangent⟫_ℝ = 0)
    (hb : ‖backwardTangent‖ = 1)
    (hcurrent : 0 < currentTurn) (hprevious : 0 < previousTurn)
    (hcoherence : 0 < coherence) :
    ⟪cplgCandidate u backwardTangent currentTurn previousTurn coherence,
        backwardTangent⟫_ℝ <
      ⟪u, backwardTangent⟫_ℝ := by
  have hinner : 0 < min currentTurn previousTurn := by
    exact lt_min hcurrent hprevious
  have hturn : 0 < min (min currentTurn previousTurn) (1 / 4 : ℝ) := by
    exact lt_min hinner (by norm_num)
  have hphase : phaseLock coherence = coherence := by
    exact max_eq_right hcoherence.le
  have hratio : 0 < cplgRatio currentTurn previousTurn coherence := by
    rw [cplgRatio, hphase]
    exact mul_pos hcoherence hturn
  exact cgc_reversal_alignment_lt u backwardTangent
    (cplgRatio currentTurn previousTurn coherence) hub hb hratio

end

end LeanMechanism
