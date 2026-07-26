import Mathlib
import LeanMechanism.QuadraticAlignment

/-!
# The variance cost of finite-horizon bias correction

This file works at the **scalar variance level**.  For independent, zero-mean
inputs `xi_k` with common variance `sigmaSq`, the raw geometric buffer at
zero-indexed round `t` has variance

`sigmaSq * sum_(k=0)^t mu^(2k)`.

No probability space is constructed: independence and zero mean enter exactly
through the standard fact that the cross-covariance terms vanish, and the
resulting variance polynomial is the model definition below.  Dividing the
random update by `1 - mu^(t+1)` multiplies its variance by the square of that
factor.  The result therefore applies to any scalar random update with the same
raw variance, but it makes no tail, concentration, or Gaussianity claim.

Indexing is important.  Lean's `t = 0` is the human **first step**, so its
factor is `1/(1-mu)^2`.  Calling that human step `t = 1` while retaining the
formula `1-mu^(t+1)` would instead produce `1/(1-mu^2)^2`.
-/

namespace LeanMechanism

noncomputable section

open Filter Finset
open scoped BigOperators Topology

/-- Raw variance of the zero-indexed finite geometric buffer under the scalar
iid, zero-mean, common-variance surrogate. -/
def rawGeometricNoiseVariance (t : Nat) (mu sigmaSq : Real) : Real :=
  sigmaSq * geometricPrefix (t + 1) (mu ^ 2)

/-- Variance multiplier introduced solely by dividing an update by
`1 - mu^(t+1)`. -/
def correctionVarianceAmplification (t : Nat) (mu : Real) : Real :=
  biasCorrectionScale t mu ^ 2

/-- Variance of the corrected scalar update. -/
def correctedGeometricNoiseVariance (t : Nat) (mu sigmaSq : Real) : Real :=
  correctionVarianceAmplification t mu *
    rawGeometricNoiseVariance t mu sigmaSq

/-- The variance-level iid calculation has the familiar finite geometric
closed form. -/
theorem rawGeometricNoiseVariance_closed_form
    (t : Nat) (mu sigmaSq : Real) (hmu2 : mu ^ 2 ≠ 1) :
    rawGeometricNoiseVariance t mu sigmaSq =
      sigmaSq * (1 - (mu ^ 2) ^ (t + 1)) / (1 - mu ^ 2) := by
  rw [rawGeometricNoiseVariance,
    geometricPrefix_eq_div (t + 1) (mu ^ 2) hmu2]
  ring

/-- Exact cost-of-correction identity: variance scales by the square of the
same scalar that scales the random update. -/
theorem correctedVariance_eq_amplification_mul_raw
    (t : Nat) (mu sigmaSq : Real) :
    correctedGeometricNoiseVariance t mu sigmaSq =
      correctionVarianceAmplification t mu *
        rawGeometricNoiseVariance t mu sigmaSq := rfl

/-- In the positive-variance stable regime, the corrected/raw variance ratio
is exactly `1/(1-mu^(t+1))^2`. -/
theorem corrected_to_raw_variance_ratio
    (t : Nat) (mu sigmaSq : Real)
    (_hmu0 : 0 ≤ mu) (_hmu1 : mu < 1) (hsigma : 0 < sigmaSq) :
    correctedGeometricNoiseVariance t mu sigmaSq /
        rawGeometricNoiseVariance t mu sigmaSq =
      1 / (1 - mu ^ (t + 1)) ^ 2 := by
  have hprefix0 : 0 ≤ geometricPrefix t (mu ^ 2) := by
    unfold geometricPrefix
    exact sum_nonneg fun k _hk => pow_nonneg (sq_nonneg mu) k
  have hprefix : 0 < geometricPrefix (t + 1) (mu ^ 2) := by
    rw [geometricPrefix_succ]
    positivity
  have hraw : rawGeometricNoiseVariance t mu sigmaSq ≠ 0 := by
    unfold rawGeometricNoiseVariance
    positivity
  rw [correctedVariance_eq_amplification_mul_raw]
  calc
    correctionVarianceAmplification t mu *
          rawGeometricNoiseVariance t mu sigmaSq /
        rawGeometricNoiseVariance t mu sigmaSq =
        correctionVarianceAmplification t mu := by
          rw [mul_comm, mul_div_cancel_left₀ _ hraw]
    _ = 1 / (1 - mu ^ (t + 1)) ^ 2 := by
      unfold correctionVarianceAmplification biasCorrectionScale
      simp only [one_div, inv_pow]

/-- On the human first step (`t = 0` in the formula), the exact amplification
is the requested maximal factor `1/(1-mu)^2`. -/
theorem correctionVarianceAmplification_first_step (mu : Real) :
    correctionVarianceAmplification 0 mu = 1 / (1 - mu) ^ 2 := by
  unfold correctionVarianceAmplification biasCorrectionScale
  simp

/-- The corrected first-step variance itself is
`sigmaSq/(1-mu)^2`. -/
theorem correctedGeometricNoiseVariance_first_step
    (mu sigmaSq : Real) :
    correctedGeometricNoiseVariance 0 mu sigmaSq =
      sigmaSq / (1 - mu) ^ 2 := by
  unfold correctedGeometricNoiseVariance correctionVarianceAmplification
    rawGeometricNoiseVariance biasCorrectionScale
  simp [geometricPrefix]
  ring

/-- The correction's variance multiplier decreases with zero-indexed step age
for `0 ≤ mu < 1`. -/
theorem correctionVarianceAmplification_antitone
    (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Antitone (fun t : Nat => correctionVarianceAmplification t mu) := by
  apply antitone_nat_of_succ_le
  intro t
  have hpow_old : mu ^ (t + 1) < 1 :=
    pow_lt_one₀ hmu0 hmu1 (by omega)
  have hpow_new : mu ^ (t + 2) < 1 :=
    pow_lt_one₀ hmu0 hmu1 (by omega)
  have hpow_le : mu ^ (t + 2) ≤ mu ^ (t + 1) := by
    rw [show t + 2 = (t + 1) + 1 by omega, pow_succ]
    exact mul_le_of_le_one_right (pow_nonneg hmu0 (t + 1)) hmu1.le
  have hden_le : 1 - mu ^ (t + 1) ≤ 1 - mu ^ (t + 2) := by
    linarith
  have hscale :
      1 / (1 - mu ^ (t + 2)) ≤ 1 / (1 - mu ^ (t + 1)) :=
    one_div_le_one_div_of_le (sub_pos.mpr hpow_old) hden_le
  unfold correctionVarianceAmplification biasCorrectionScale
  simp only [Nat.add_assoc, Nat.reduceAdd]
  exact (sq_le_sq₀
    (one_div_nonneg.mpr (sub_nonneg.mpr hpow_new.le))
    (one_div_nonneg.mpr (sub_nonneg.mpr hpow_old.le))).2 hscale

/-- Hence the human first step is globally maximal among all later steps. -/
theorem correctionVarianceAmplification_le_first_step
    (t : Nat) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    correctionVarianceAmplification t mu ≤
      correctionVarianceAmplification 0 mu :=
  correctionVarianceAmplification_antitone mu hmu0 hmu1 (Nat.zero_le t)

/-- The variance amplification decays to its steady ratio `1`: the correction
becomes asymptotically variance-neutral relative to the raw update. -/
theorem correctionVarianceAmplification_tendsto_one
    (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Tendsto (fun t : Nat => correctionVarianceAmplification t mu)
      atTop (nhds 1) := by
  have hp : Tendsto (fun t : Nat => mu ^ t) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hmu0 hmu1
  have hp' : Tendsto (fun t : Nat => mu ^ (t + 1)) atTop (nhds 0) :=
    hp.comp (tendsto_add_atTop_nat 1)
  have hden : Tendsto (fun t : Nat => 1 - mu ^ (t + 1)) atTop (nhds 1) := by
    simpa using tendsto_const_nhds.sub hp'
  have hscale : Tendsto (fun t : Nat => biasCorrectionScale t mu)
      atTop (nhds 1) := by
    simpa [biasCorrectionScale, one_div] using
      hden.inv₀ (by norm_num : (1 : Real) ≠ 0)
  simpa [correctionVarianceAmplification] using hscale.pow 2

/-- The raw iid-geometric variance converges to
`sigmaSq/(1-mu^2)`. -/
theorem rawGeometricNoiseVariance_tendsto
    (mu sigmaSq : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Tendsto (fun t : Nat => rawGeometricNoiseVariance t mu sigmaSq)
      atTop (nhds (sigmaSq / (1 - mu ^ 2))) := by
  have hmu2_0 : 0 ≤ mu ^ 2 := sq_nonneg mu
  have hmu2_1 : mu ^ 2 < 1 := by nlinarith [sq_nonneg (1 - mu)]
  have hgeo :
      Tendsto (fun T : Nat => geometricPrefix T (mu ^ 2)) atTop
        (nhds (1 / (1 - mu ^ 2))) := by
    simpa [terminalMultiplier, steadyMultiplier] using
      (terminalMultiplier_tendsto_steady
        OuterRule.heavyBall (mu ^ 2) hmu2_0 hmu2_1)
  have hgeo' := hgeo.comp (tendsto_add_atTop_nat 1)
  simpa [rawGeometricNoiseVariance, div_eq_mul_inv] using
    (tendsto_const_nhds.mul hgeo' :
      Tendsto
        (fun t : Nat => sigmaSq * geometricPrefix (t + 1) (mu ^ 2))
        atTop (nhds (sigmaSq * (1 / (1 - mu ^ 2)))))

/-- Corrected and raw variances share the same steady limit, because their
ratio tends to one. -/
theorem correctedGeometricNoiseVariance_tendsto
    (mu sigmaSq : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Tendsto (fun t : Nat => correctedGeometricNoiseVariance t mu sigmaSq)
      atTop (nhds (sigmaSq / (1 - mu ^ 2))) := by
  simpa [correctedGeometricNoiseVariance] using
    (correctionVarianceAmplification_tendsto_one mu hmu0 hmu1).mul
      (rawGeometricNoiseVariance_tendsto mu sigmaSq hmu0 hmu1)

/-- Mean-side benefit of the same correction in the constant-input model:
every age has exactly the steady multiplier.  Together with the theorems above,
this states the trade honestly: early-step variance buys horizon-invariance of
the mean step. -/
theorem correction_mean_multiplier_horizon_invariant
    (t : Nat) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    correctedTerminalMultiplier t mu = steadyMultiplier mu :=
  correctedTerminalMultiplier_eq_steady t mu hmu0 hmu1

end

end LeanMechanism
