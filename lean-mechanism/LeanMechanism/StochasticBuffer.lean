import Mathlib
import LeanMechanism.FiniteHorizonOuter

/-!
# Correlation-limited momentum-buffer saturation

This file uses a **scalar second-moment surrogate**, rather than constructing a
probability space of vector-valued random variables.  The modeled inputs have
unit second moment and geometric lag covariance

`E[g_t g_(t-k)] = rho^k`.

For `b_(t+1) = mu * b_t + g_(t+1)`, `correlatedCrossMoment t mu rho` is the
modeled cross moment `E[b_t g_(t+1)]`, and
`finiteCorrelatedBufferSecondMoment t mu rho` is `E[b_t^2]`.  Thus the square
root is an RMS norm gain.  This proves an exact statement for the scalar model;
it also applies coordinatewise (and hence after summing coordinates) whenever
the vector covariance kernel is isotropic with the same `rho`.  It does not
claim that arbitrary training deltas have an exactly geometric covariance
kernel.
-/

namespace LeanMechanism

noncomputable section

open Filter Finset
open scoped BigOperators Topology

/-- Under the geometric covariance kernel, the finite cross moment
`E[b_T g_(T+1)] = rho * (1 + mu*rho + ... + (mu*rho)^(T-1))`. -/
def correlatedCrossMoment (T : Nat) (mu rho : Real) : Real :=
  rho * geometricPrefix T (mu * rho)

private def finiteKernelSecondMoment (a r : Real) : Nat → Real
  | 0 => 0
  | T + 1 =>
      a * finiteKernelSecondMoment a r T + 1 +
        2 * r * geometricPrefix T r

/-- Exact finite-time second-moment recursion for the scalar surrogate.

The three terms are respectively the retained buffer variance, the unit input
variance, and twice the buffer/input covariance.  This recursive definition is
the finite-`T` truncation and is valid without division or asymptotics.
-/
def finiteCorrelatedBufferSecondMoment (mu rho : Real) (T : Nat) : Real :=
  finiteKernelSecondMoment (mu ^ 2) (mu * rho) T

/-- Finite-time RMS buffer gain in the scalar second-moment model. -/
def finiteCorrelationGain (T : Nat) (mu rho : Real) : Real :=
  Real.sqrt (finiteCorrelatedBufferSecondMoment mu rho T)

/-- The stationary cross moment selected by the stable geometric kernel. -/
def stationaryCrossMoment (mu rho : Real) : Real :=
  rho / (1 - mu * rho)

/-- The stationary squared RMS gain. -/
def correlationLimitedSquaredGain (mu rho : Real) : Real :=
  (1 + mu * rho) / ((1 - mu ^ 2) * (1 - mu * rho))

/-- The correlation-limited RMS norm gain

`G(mu,rho) = sqrt ((1 + mu*rho) / ((1-mu^2)(1-mu*rho)))`.
-/
def correlationLimitedGain (mu rho : Real) : Real :=
  Real.sqrt (correlationLimitedSquaredGain mu rho)

@[simp]
theorem finiteCorrelatedBufferSecondMoment_zero (mu rho : Real) :
    finiteCorrelatedBufferSecondMoment mu rho 0 = 0 := rfl

/-- Exact finite-step truncation recurrence. -/
@[simp]
theorem finiteCorrelatedBufferSecondMoment_succ (T : Nat) (mu rho : Real) :
    finiteCorrelatedBufferSecondMoment mu rho (T + 1) =
      mu ^ 2 * finiteCorrelatedBufferSecondMoment mu rho T + 1 +
        2 * mu * correlatedCrossMoment T mu rho := by
  simp only [finiteCorrelatedBufferSecondMoment, finiteKernelSecondMoment,
    correlatedCrossMoment]
  ring

/-- The modeled cross moment itself obeys the AR(1) covariance recursion. -/
theorem correlatedCrossMoment_succ (T : Nat) (mu rho : Real) :
    correlatedCrossMoment (T + 1) mu rho =
      mu * rho * correlatedCrossMoment T mu rho + rho := by
  rw [correlatedCrossMoment, correlatedCrossMoment, geometricPrefix_succ]
  ring

/-- The first buffer contains one unit-variance input. -/
@[simp]
theorem finiteCorrelatedBufferSecondMoment_one (mu rho : Real) :
    finiteCorrelatedBufferSecondMoment mu rho 1 = 1 := by
  simp [correlatedCrossMoment]

/-- The two-step truncation exposes the first correlation contribution. -/
theorem finiteCorrelatedBufferSecondMoment_two (mu rho : Real) :
    finiteCorrelatedBufferSecondMoment mu rho 2 =
      1 + mu ^ 2 + 2 * mu * rho := by
  simp [correlatedCrossMoment, geometricPrefix]
  ring

private theorem finiteKernelSecondMoment_closed_form
    (T : Nat) (a r : Real) (ha : a ≠ 1) (hr : r ≠ 1) (har : a ≠ r) :
    finiteKernelSecondMoment a r T =
      ((1 + r) / ((1 - a) * (1 - r))) * (1 - a ^ T) -
        (2 * r / (1 - r)) * ((a ^ T - r ^ T) / (a - r)) := by
  have haden : 1 - a ≠ 0 := sub_ne_zero.mpr (Ne.symm ha)
  have hrden : 1 - r ≠ 0 := sub_ne_zero.mpr (Ne.symm hr)
  have harden : a - r ≠ 0 := sub_ne_zero.mpr har
  induction T with
  | zero => simp [finiteKernelSecondMoment]
  | succ T ih =>
      rw [finiteKernelSecondMoment, ih, geometricPrefix_eq_div T r hr]
      rw [show a ^ (T + 1) = a ^ T * a by exact pow_succ a T]
      rw [show r ^ (T + 1) = r ^ T * r by exact pow_succ r T]
      field_simp [haden, hrden, harden]
      ring

/-- A closed finite-`T` formula away from the removable resonance
`mu^2 = mu*rho`.

The second term is the finite transient caused by the correlated forcing.  The
recursive truncation above remains the definition at the resonance, so no model
case is left undefined; only this convenient quotient form excludes it.
-/
theorem finiteCorrelatedBufferSecondMoment_closed_form
    (T : Nat) (mu rho : Real)
    (hmu2 : mu ^ 2 ≠ 1) (hmurho : mu * rho ≠ 1)
    (hres : mu ^ 2 ≠ mu * rho) :
    finiteCorrelatedBufferSecondMoment mu rho T =
      correlationLimitedSquaredGain mu rho * (1 - (mu ^ 2) ^ T) -
        (2 * (mu * rho) / (1 - mu * rho)) *
          (((mu ^ 2) ^ T - (mu * rho) ^ T) / (mu ^ 2 - mu * rho)) := by
  simpa [finiteCorrelatedBufferSecondMoment,
    correlationLimitedSquaredGain] using
    finiteKernelSecondMoment_closed_form T (mu ^ 2) (mu * rho)
      hmu2 hmurho hres

/-- Under the stable nonnegative parameter regime, every finite truncation is a
nonnegative second moment, as required for the RMS interpretation. -/
theorem finiteCorrelatedBufferSecondMoment_nonneg
    (T : Nat) (mu rho : Real) (hmu : 0 ≤ mu) (hrho : 0 ≤ rho) :
    0 ≤ finiteCorrelatedBufferSecondMoment mu rho T := by
  induction T with
  | zero => simp
  | succ T ih =>
      rw [finiteCorrelatedBufferSecondMoment_succ]
      have hprefix : 0 ≤ geometricPrefix T (mu * rho) := by
        unfold geometricPrefix
        exact sum_nonneg fun k _hk => pow_nonneg (mul_nonneg hmu hrho) k
      have hcross : 0 ≤ correlatedCrossMoment T mu rho :=
        mul_nonneg hrho hprefix
      positivity

/-- The stationary cross moment is the fixed point of the geometric
cross-covariance recursion. -/
theorem stationaryCrossMoment_balance
    (mu rho : Real) (h : mu * rho ≠ 1) :
    stationaryCrossMoment mu rho =
      mu * rho * stationaryCrossMoment mu rho + rho := by
  have hden : 1 - mu * rho ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  have hden' : 1 - rho * mu ≠ 0 := by simpa [mul_comm] using hden
  unfold stationaryCrossMoment
  field_simp [hden, hden']
  ring

/-- Solving the stationary second-moment balance gives the advertised squared
gain formula.  This is the algebraic derivation of the saturation law. -/
theorem correlationLimitedSquaredGain_balance
    (mu rho : Real) (hmu2 : mu ^ 2 ≠ 1) (hmurho : mu * rho ≠ 1) :
    correlationLimitedSquaredGain mu rho =
      mu ^ 2 * correlationLimitedSquaredGain mu rho + 1 +
        2 * mu * stationaryCrossMoment mu rho := by
  have hmu2den : 1 - mu ^ 2 ≠ 0 := sub_ne_zero.mpr (Ne.symm hmu2)
  have hmurhoden : 1 - mu * rho ≠ 0 := sub_ne_zero.mpr (Ne.symm hmurho)
  unfold correlationLimitedSquaredGain stationaryCrossMoment
  field_simp [hmu2den, hmurhoden]
  ring

/-- In the stable nonnegative regime, the displayed stationary quantity is a
genuine square: `G(mu,rho)^2` equals the rational squared-gain formula. -/
theorem correlationLimitedGain_sq
    (mu rho : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1)
    (hrho0 : 0 ≤ rho) (hrho1 : rho ≤ 1) :
    correlationLimitedGain mu rho ^ 2 =
      correlationLimitedSquaredGain mu rho := by
  have hmu2 : mu ^ 2 < 1 := by nlinarith [sq_nonneg (1 - mu)]
  have hmurho_le : mu * rho ≤ mu := mul_le_of_le_one_right hmu0 hrho1
  have hmurho : mu * rho < 1 := lt_of_le_of_lt hmurho_le hmu1
  have hnum : 0 ≤ 1 + mu * rho := by positivity
  have hden : 0 < (1 - mu ^ 2) * (1 - mu * rho) :=
    mul_pos (sub_pos.mpr hmu2) (sub_pos.mpr hmurho)
  unfold correlationLimitedGain
  apply Real.sq_sqrt
  unfold correlationLimitedSquaredGain
  exact div_nonneg hnum hden.le

/-- Away from the removable resonance of the quotient-form finite transient,
the finite RMS gain converges to `G(mu,rho)`. -/
theorem finiteCorrelationGain_tendsto
    (mu rho : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1)
    (hrho0 : 0 ≤ rho) (hrho1 : rho ≤ 1)
    (hres : mu ^ 2 ≠ mu * rho) :
    Tendsto (fun T : Nat => finiteCorrelationGain T mu rho) atTop
      (nhds (correlationLimitedGain mu rho)) := by
  have hmu2_0 : 0 ≤ mu ^ 2 := sq_nonneg mu
  have hmu2_1 : mu ^ 2 < 1 := by nlinarith [sq_nonneg (1 - mu)]
  have hmurho_0 : 0 ≤ mu * rho := mul_nonneg hmu0 hrho0
  have hmurho_le : mu * rho ≤ mu := mul_le_of_le_one_right hmu0 hrho1
  have hmurho_1 : mu * rho < 1 := lt_of_le_of_lt hmurho_le hmu1
  have hmu2_ne : mu ^ 2 ≠ 1 := ne_of_lt hmu2_1
  have hmurho_ne : mu * rho ≠ 1 := ne_of_lt hmurho_1
  have ha : Tendsto (fun T : Nat => (mu ^ 2) ^ T) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hmu2_0 hmu2_1
  have hr : Tendsto (fun T : Nat => (mu * rho) ^ T) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hmurho_0 hmurho_1
  have honeSub : Tendsto (fun T : Nat => 1 - (mu ^ 2) ^ T) atTop (nhds 1) := by
    simpa using tendsto_const_nhds.sub ha
  have hquot :
      Tendsto
        (fun T : Nat =>
          (((mu ^ 2) ^ T - (mu * rho) ^ T) / (mu ^ 2 - mu * rho)))
        atTop (nhds 0) := by
    simpa using (ha.sub hr).div_const (mu ^ 2 - mu * rho)
  have hsecond :
      Tendsto
        (fun T : Nat =>
          (2 * (mu * rho) / (1 - mu * rho)) *
            (((mu ^ 2) ^ T - (mu * rho) ^ T) / (mu ^ 2 - mu * rho)))
        atTop (nhds 0) := by
    simpa using
      (tendsto_const_nhds.mul hquot :
        Tendsto
          (fun T : Nat =>
            (2 * (mu * rho) / (1 - mu * rho)) *
              (((mu ^ 2) ^ T - (mu * rho) ^ T) / (mu ^ 2 - mu * rho)))
          atTop (nhds ((2 * (mu * rho) / (1 - mu * rho)) * 0)))
  have hsq :
      Tendsto
        (fun T : Nat => finiteCorrelatedBufferSecondMoment mu rho T)
        atTop (nhds (correlationLimitedSquaredGain mu rho)) := by
    have hfirst :
        Tendsto
          (fun T : Nat =>
            correlationLimitedSquaredGain mu rho * (1 - (mu ^ 2) ^ T))
          atTop (nhds (correlationLimitedSquaredGain mu rho)) := by
      simpa using
        (tendsto_const_nhds.mul honeSub :
          Tendsto
            (fun T : Nat =>
              correlationLimitedSquaredGain mu rho * (1 - (mu ^ 2) ^ T))
            atTop (nhds (correlationLimitedSquaredGain mu rho * 1)))
    refine Tendsto.congr'
      (f₁ := fun T : Nat =>
        correlationLimitedSquaredGain mu rho * (1 - (mu ^ 2) ^ T) -
          (2 * (mu * rho) / (1 - mu * rho)) *
            (((mu ^ 2) ^ T - (mu * rho) ^ T) / (mu ^ 2 - mu * rho)))
      (Eventually.of_forall fun T => ?_) ?_
    · exact (finiteCorrelatedBufferSecondMoment_closed_form
        T mu rho hmu2_ne hmurho_ne hres).symm
    · simpa using hfirst.sub hsecond
  exact hsq.sqrt

/-- Perfect correlation recovers the familiar coherent gain
`G(mu,1) = 1/(1-mu)`. -/
theorem correlationLimitedGain_rho_one
    (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    correlationLimitedGain mu 1 = 1 / (1 - mu) := by
  have hden : 0 < 1 - mu := sub_pos.mpr hmu1
  have hne : 1 - mu ≠ 0 := hden.ne'
  have hmu2 : mu ^ 2 < 1 := by nlinarith [sq_nonneg (1 - mu)]
  have hmu2den : 1 - mu ^ 2 ≠ 0 := (sub_pos.mpr hmu2).ne'
  have hsq : correlationLimitedSquaredGain mu 1 = (1 / (1 - mu)) ^ 2 := by
    unfold correlationLimitedSquaredGain
    field_simp [hne, hmu2den]
    ring
  rw [correlationLimitedGain, hsq, Real.sqrt_sq_eq_abs,
    abs_of_pos (one_div_pos.mpr hden)]

/-- Zero lag correlation gives the independent-input RMS gain
`G(mu,0) = 1/sqrt(1-mu^2)`. -/
theorem correlationLimitedGain_rho_zero
    (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    correlationLimitedGain mu 0 = 1 / Real.sqrt (1 - mu ^ 2) := by
  have hmu2 : 0 ≤ 1 - mu ^ 2 := by
    nlinarith [sq_nonneg (1 - mu)]
  unfold correlationLimitedGain correlationLimitedSquaredGain
  rw [show (1 + mu * 0) / ((1 - mu ^ 2) * (1 - mu * 0)) =
      (1 : Real) / (1 - mu ^ 2) by ring]
  rw [Real.sqrt_div (by norm_num : (0 : Real) ≤ 1)]
  simp

/-- For fixed stable nonnegative `mu`, correlation-limited RMS gain is monotone
in `rho` on the modeled correlation range `[0,1]`. -/
theorem correlationLimitedGain_monotoneOn
    (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    MonotoneOn (correlationLimitedGain mu) (Set.Icc (0 : Real) 1) := by
  intro rho₁ hrho₁ rho₂ hrho₂ hle
  unfold correlationLimitedGain
  apply Real.sqrt_le_sqrt
  unfold correlationLimitedSquaredGain
  have hmu2 : mu ^ 2 < 1 := by nlinarith [sq_nonneg (1 - mu)]
  have hA : 0 < 1 - mu ^ 2 := sub_pos.mpr hmu2
  have hr1le : mu * rho₁ ≤ mu := mul_le_of_le_one_right hmu0 hrho₁.2
  have hr2le : mu * rho₂ ≤ mu := mul_le_of_le_one_right hmu0 hrho₂.2
  have hd1 : 0 < (1 - mu ^ 2) * (1 - mu * rho₁) :=
    mul_pos hA (sub_pos.mpr (lt_of_le_of_lt hr1le hmu1))
  have hd2 : 0 < (1 - mu ^ 2) * (1 - mu * rho₂) :=
    mul_pos hA (sub_pos.mpr (lt_of_le_of_lt hr2le hmu1))
  apply (div_le_div_iff₀ hd1 hd2).2
  nlinarith [mul_nonneg hmu0 (sub_nonneg.mpr hle)]

/-- Consequently, the independent and perfectly correlated laws bracket every
nonnegative geometric-correlation gain. -/
theorem correlationLimitedGain_between_extremes
    (mu rho : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1)
    (hrho0 : 0 ≤ rho) (hrho1 : rho ≤ 1) :
    1 / Real.sqrt (1 - mu ^ 2) ≤ correlationLimitedGain mu rho ∧
      correlationLimitedGain mu rho ≤ 1 / (1 - mu) := by
  have hmono := correlationLimitedGain_monotoneOn mu hmu0 hmu1
  constructor
  · rw [← correlationLimitedGain_rho_zero mu hmu0 hmu1]
    exact hmono ⟨le_rfl, zero_le_one⟩ ⟨hrho0, hrho1⟩ hrho0
  · rw [← correlationLimitedGain_rho_one mu hmu0 hmu1]
    exact hmono ⟨hrho0, hrho1⟩ ⟨zero_le_one, le_rfl⟩ hrho1

end

end LeanMechanism
