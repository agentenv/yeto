import LeanMechanism.TransferPenalty
import LeanMechanism.StochasticBuffer
import LeanMechanism.CorrectionCosts

/-!
# Scalar noise phase boundaries

This module extends the frozen-gradient scalar quadratic with a second-moment
noise term.  It proves two related, but deliberately distinct, boundaries.

## Exactly retuned boundary

At zero-indexed update `t`, the heavy-ball direction has coherent mean
multiplier `geometricPrefix (t+1) mu`.  Its iid scalar buffer-noise variance is
`sigmaSq * geometricPrefix (t+1) (mu^2)`, exactly the marginal variance from
`CorrectionCosts` and the `rho = 0` case of `StochasticBuffer`.  Assuming these
outer-update direction noises have zero cross-covariance, expected terminal
loss is represented by `iidNoisyFrozenRisk`.

The displayed `iidNoisyOptimalEta` is proved to be the global optimum.  After
both momentum and no momentum are retuned, the helps/ties/hurts boundary is

`c(T,mu)^2` compared with `T * v(T,mu)`.

For positive noise, momentum helps iff the left side is larger, ties iff equal,
and hurts iff smaller.  The magnitude `sigmaSq / (a^2 * theta0^2)` cancels in
this homogeneous one-dimensional model.  Thus a noise-scale-dependent phase
diagram cannot be obtained from exact retuning in this surrogate alone.

## Finite steady-prescription boundary

The asymptotic prescription `(1-mu) * eta0(T)` is not the exact finite-horizon
rate.  Its deterministic shortfall competes with its iid variance saving.  For
this prescription the phase boundary does depend on the dimensionless
noise-to-signal ratio `sigmaSq / (a^2 * theta0^2)`, and an exact threshold is
proved below.

## Model scope and the `T = 5` observation

No probability space, tail bound, nonlinear trajectory, vector anisotropy, or
cross-update covariance is constructed.  Only the stated scalar second moments
are modeled.  In particular, persistent buffer noise normally creates
cross-update covariance; setting those terms to zero is an explicit surrogate
assumption, not a theorem about the production optimizer.

At `T = 5` the iid exactly-retuned scalar model has positive coherence advantage
for `0 < mu < 1`, so it predicts HELP rather than the paper's `SNOO_HURTS`.
That is an honest qualitative mismatch and points to omitted correlation,
evolving gradients, anisotropy, or implementation effects.  The finite steady
prescription can predict HURTS at `T = 5` precisely on the low-noise side of the
proved threshold; no claim is made here that the experiment's unmodeled
quantities satisfy that inequality.
-/

namespace LeanMechanism

noncomputable section

open Finset
open scoped BigOperators

/-! ## Noise factors and links to lane 4 -/

/-- Sum of iid geometric-buffer marginal variance multipliers over `T`
heavy-ball updates. -/
def iidOuterNoiseFactor (T : Nat) (mu : Real) : Real :=
  effectiveCoeff .heavyBall T (mu ^ 2)

/-- The factor is exactly the sum of the per-step raw variances from
`CorrectionCosts`, after pulling out `sigmaSq`. -/
theorem sum_rawGeometricNoiseVariance_eq
    (T : Nat) (mu sigmaSq : Real) :
    (∑ t ∈ range T, rawGeometricNoiseVariance t mu sigmaSq) =
      sigmaSq * iidOuterNoiseFactor T mu := by
  simp [rawGeometricNoiseVariance, iidOuterNoiseFactor, effectiveCoeff,
    terminalMultiplier, mul_sum]

/-- The independent-input (`rho = 0`) stochastic-buffer second moment is the
same finite geometric prefix used by `CorrectionCosts`. -/
theorem finiteCorrelatedBufferSecondMoment_rho_zero
    (T : Nat) (mu : Real) :
    finiteCorrelatedBufferSecondMoment mu 0 T = geometricPrefix T (mu ^ 2) := by
  induction T with
  | zero => simp
  | succ T ih =>
      rw [finiteCorrelatedBufferSecondMoment_succ, ih, geometricPrefix_succ]
      simp [correlatedCrossMoment]

/-- Equivalent expression through the `StochasticBuffer` machinery. -/
theorem iidOuterNoiseFactor_eq_sum_stochasticBuffer
    (T : Nat) (mu : Real) :
    iidOuterNoiseFactor T mu =
      ∑ t ∈ range T,
        finiteCorrelatedBufferSecondMoment mu 0 (t + 1) := by
  unfold iidOuterNoiseFactor effectiveCoeff
  simp_rw [finiteCorrelatedBufferSecondMoment_rho_zero]
  rfl

theorem iidOuterNoiseFactor_nonneg (T : Nat) (mu : Real) :
    0 <= iidOuterNoiseFactor T mu := by
  unfold iidOuterNoiseFactor effectiveCoeff
  exact sum_nonneg fun t _ht => by
    unfold terminalMultiplier geometricPrefix
    exact sum_nonneg fun k _hk => pow_nonneg (sq_nonneg mu) k

theorem iidOuterNoiseFactor_pos
    (T : Nat) (mu : Real) (hT : 0 < T) :
    0 < iidOuterNoiseFactor T mu :=
  effectiveCoeff_pos .heavyBall T (mu ^ 2) hT (sq_nonneg mu)

@[simp]
theorem iidOuterNoiseFactor_zero_momentum (T : Nat) :
    iidOuterNoiseFactor T 0 = T := by
  simp [iidOuterNoiseFactor]

/-! ## A generic scalar bias-variance objective -/

/-- Expected terminal-loss surrogate for a frozen heavy-ball mean path and a
supplied dimensionless total noise factor `v`. -/
def heavyBallSecondMomentRisk
    (T : Nat) (mu eta theta0 a sigmaSq v : Real) : Real :=
  frozenQuadraticJ .heavyBall T mu eta theta0 a +
    (a / 2) * eta ^ 2 * sigmaSq * v

/-- Positive quadratic coefficient in the noisy optimization problem. -/
def heavyBallSecondMomentDenom
    (T : Nat) (mu theta0 a sigmaSq v : Real) : Real :=
  a ^ 2 * theta0 ^ 2 * effectiveCoeff .heavyBall T mu ^ 2 + sigmaSq * v

/-- Closed optimizer of the scalar second-moment risk. -/
def heavyBallSecondMomentOptimalEta
    (T : Nat) (mu theta0 a sigmaSq v : Real) : Real :=
  a * theta0 ^ 2 * effectiveCoeff .heavyBall T mu /
    heavyBallSecondMomentDenom T mu theta0 a sigmaSq v

/-- Exact completion of the square. -/
theorem heavyBallSecondMomentRisk_eq_min_add_square
    (T : Nat) (mu eta theta0 a sigmaSq v : Real)
    (hden : heavyBallSecondMomentDenom T mu theta0 a sigmaSq v ≠ 0) :
    heavyBallSecondMomentRisk T mu eta theta0 a sigmaSq v =
      (a / 2) *
        (theta0 ^ 2 * sigmaSq * v /
            heavyBallSecondMomentDenom T mu theta0 a sigmaSq v +
          heavyBallSecondMomentDenom T mu theta0 a sigmaSq v *
            (eta - heavyBallSecondMomentOptimalEta
              T mu theta0 a sigmaSq v) ^ 2) := by
  have hden' :
      a ^ 2 * theta0 ^ 2 * effectiveCoeff .heavyBall T mu ^ 2 + sigmaSq * v ≠ 0 := by
    simpa [heavyBallSecondMomentDenom] using hden
  unfold heavyBallSecondMomentRisk frozenQuadraticJ
    frozenGradientQuadraticLoss heavyBallSecondMomentOptimalEta
    heavyBallSecondMomentDenom
  field_simp [hden']
  ring

/-- The noisy denominator is positive under the scalar model assumptions. -/
theorem heavyBallSecondMomentDenom_pos
    (T : Nat) (mu theta0 a sigmaSq v : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (htheta : theta0 ≠ 0)
    (ha : 0 < a) (hsigma : 0 <= sigmaSq) (hv : 0 <= v) :
    0 < heavyBallSecondMomentDenom T mu theta0 a sigmaSq v := by
  have hc : 0 < effectiveCoeff .heavyBall T mu :=
    effectiveCoeff_pos .heavyBall T mu hT hmu
  unfold heavyBallSecondMomentDenom
  have hsignal :
      0 < a ^ 2 * theta0 ^ 2 * effectiveCoeff .heavyBall T mu ^ 2 := by
    positivity
  nlinarith [mul_nonneg hsigma hv]

/-- The displayed rate is a global minimizer of the noisy scalar surrogate. -/
theorem heavyBallSecondMomentOptimalEta_is_minimizer
    (T : Nat) (mu theta0 a sigmaSq v : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (htheta : theta0 ≠ 0)
    (ha : 0 < a) (hsigma : 0 <= sigmaSq) (hv : 0 <= v) :
    forall eta : Real,
      heavyBallSecondMomentRisk T mu
          (heavyBallSecondMomentOptimalEta T mu theta0 a sigmaSq v)
          theta0 a sigmaSq v <=
        heavyBallSecondMomentRisk T mu eta theta0 a sigmaSq v := by
  intro eta
  have hden := heavyBallSecondMomentDenom_pos T mu theta0 a sigmaSq v
    hT hmu htheta ha hsigma hv
  rw [heavyBallSecondMomentRisk_eq_min_add_square T mu eta theta0 a sigmaSq v hden.ne',
    heavyBallSecondMomentRisk_eq_min_add_square T mu
      (heavyBallSecondMomentOptimalEta T mu theta0 a sigmaSq v)
      theta0 a sigmaSq v hden.ne']
  simp only [sub_self, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true,
    zero_pow, mul_zero, add_zero, ge_iff_le]
  apply mul_le_mul_of_nonneg_left _ (by positivity)
  exact le_add_of_nonneg_right (mul_nonneg hden.le (sq_nonneg _))

/-- Closed value at the noisy optimum. -/
theorem heavyBallSecondMomentRisk_at_optimal
    (T : Nat) (mu theta0 a sigmaSq v : Real)
    (hden : heavyBallSecondMomentDenom T mu theta0 a sigmaSq v ≠ 0) :
    heavyBallSecondMomentRisk T mu
        (heavyBallSecondMomentOptimalEta T mu theta0 a sigmaSq v)
        theta0 a sigmaSq v =
      (a / 2) * theta0 ^ 2 * sigmaSq * v /
        heavyBallSecondMomentDenom T mu theta0 a sigmaSq v := by
  rw [heavyBallSecondMomentRisk_eq_min_add_square T mu
    (heavyBallSecondMomentOptimalEta T mu theta0 a sigmaSq v)
    theta0 a sigmaSq v hden]
  simp [heavyBallSecondMomentOptimalEta]
  ring

/-! ## Exactly retuned iid phase boundary -/

/-- The iid specialization of the second-moment risk. -/
def iidNoisyFrozenRisk
    (T : Nat) (mu eta theta0 a sigmaSq : Real) : Real :=
  heavyBallSecondMomentRisk T mu eta theta0 a sigmaSq
    (iidOuterNoiseFactor T mu)

/-- Exact noisy optimum in the iid marginal-variance surrogate. -/
def iidNoisyOptimalEta
    (T : Nat) (mu theta0 a sigmaSq : Real) : Real :=
  heavyBallSecondMomentOptimalEta T mu theta0 a sigmaSq
    (iidOuterNoiseFactor T mu)

/-- Expected risk after exact scalar retuning. -/
def tunedIidNoisyRisk
    (T : Nat) (mu theta0 a sigmaSq : Real) : Real :=
  iidNoisyFrozenRisk T mu (iidNoisyOptimalEta T mu theta0 a sigmaSq)
    theta0 a sigmaSq

/-- The iid-specialized displayed rate is the true global optimum of the scalar
second-moment surrogate. -/
theorem iidNoisyOptimalEta_is_minimizer
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (htheta : theta0 ≠ 0)
    (ha : 0 < a) (hsigma : 0 <= sigmaSq) :
    forall eta : Real,
      iidNoisyFrozenRisk T mu (iidNoisyOptimalEta T mu theta0 a sigmaSq)
          theta0 a sigmaSq <=
        iidNoisyFrozenRisk T mu eta theta0 a sigmaSq := by
  exact heavyBallSecondMomentOptimalEta_is_minimizer T mu theta0 a sigmaSq
    (iidOuterNoiseFactor T mu) hT hmu htheta ha hsigma
    (iidOuterNoiseFactor_nonneg T mu)

/-- Coherent mean-squared advantage over the accumulated iid variance.  Its
sign is the exactly retuned phase boundary. -/
def iidCoherenceAdvantage (T : Nat) (mu : Real) : Real :=
  effectiveCoeff .heavyBall T mu ^ 2 -
    (T : Real) * iidOuterNoiseFactor T mu

private def iidNoisyDenom
    (T : Nat) (mu theta0 a sigmaSq : Real) : Real :=
  heavyBallSecondMomentDenom T mu theta0 a sigmaSq
    (iidOuterNoiseFactor T mu)

private def iidTunedGapScale
    (T : Nat) (mu theta0 a sigmaSq : Real) : Real :=
  ((a / 2) * theta0 ^ 2 * sigmaSq * (a ^ 2 * theta0 ^ 2) * (T : Real)) /
    (iidNoisyDenom T mu theta0 a sigmaSq *
      iidNoisyDenom T 0 theta0 a sigmaSq)

private theorem iidNoisyDenom_pos
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (htheta : theta0 ≠ 0)
    (ha : 0 < a) (hsigma : 0 <= sigmaSq) :
    0 < iidNoisyDenom T mu theta0 a sigmaSq := by
  exact heavyBallSecondMomentDenom_pos T mu theta0 a sigmaSq
    (iidOuterNoiseFactor T mu) hT hmu htheta ha hsigma
    (iidOuterNoiseFactor_nonneg T mu)

/-- Exact risk-gap factorization.  Every parameter outside the coherence
advantage is positive when `sigmaSq > 0`. -/
theorem tunedIidNoisyRisk_gap_exact
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (htheta : theta0 ≠ 0)
    (ha : 0 < a) (hsigma : 0 <= sigmaSq) :
    tunedIidNoisyRisk T mu theta0 a sigmaSq -
        tunedIidNoisyRisk T 0 theta0 a sigmaSq =
      -(iidTunedGapScale T mu theta0 a sigmaSq *
        iidCoherenceAdvantage T mu) := by
  have hDmu := iidNoisyDenom_pos T mu theta0 a sigmaSq
    hT hmu htheta ha hsigma
  have hDzero := iidNoisyDenom_pos T 0 theta0 a sigmaSq
    hT (by norm_num) htheta ha hsigma
  have hDmu' :
      heavyBallSecondMomentDenom T mu theta0 a sigmaSq
          (iidOuterNoiseFactor T mu) ≠ 0 := by
    simpa [iidNoisyDenom] using hDmu.ne'
  have hDzero' :
      heavyBallSecondMomentDenom T 0 theta0 a sigmaSq
          (iidOuterNoiseFactor T 0) ≠ 0 := by
    simpa [iidNoisyDenom] using hDzero.ne'
  have hDzeroT :
      heavyBallSecondMomentDenom T 0 theta0 a sigmaSq (T : Real) ≠ 0 := by
    simpa using hDzero'
  unfold tunedIidNoisyRisk iidNoisyFrozenRisk iidNoisyOptimalEta
  rw [heavyBallSecondMomentRisk_at_optimal T mu theta0 a sigmaSq
      (iidOuterNoiseFactor T mu) hDmu.ne',
    heavyBallSecondMomentRisk_at_optimal T 0 theta0 a sigmaSq
      (iidOuterNoiseFactor T 0) hDzero.ne']
  unfold iidTunedGapScale iidNoisyDenom iidCoherenceAdvantage
  simp only [iidOuterNoiseFactor_zero_momentum]
  field_simp [hDmu', hDzeroT]
  unfold heavyBallSecondMomentDenom
  rw [effectiveCoeff_zero_momentum]
  ring

private theorem iidTunedGapScale_pos
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (htheta : theta0 ≠ 0)
    (ha : 0 < a) (hsigma : 0 < sigmaSq) :
    0 < iidTunedGapScale T mu theta0 a sigmaSq := by
  have hDmu := iidNoisyDenom_pos T mu theta0 a sigmaSq
    hT hmu htheta ha hsigma.le
  have hDzero := iidNoisyDenom_pos T 0 theta0 a sigmaSq
    hT (by norm_num) htheta ha hsigma.le
  unfold iidTunedGapScale
  positivity

/-- **Exactly retuned HELP criterion.** -/
theorem tunedMomentum_helps_iff
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (htheta : theta0 ≠ 0)
    (ha : 0 < a) (hsigma : 0 < sigmaSq) :
    tunedIidNoisyRisk T mu theta0 a sigmaSq <
        tunedIidNoisyRisk T 0 theta0 a sigmaSq <->
      0 < iidCoherenceAdvantage T mu := by
  have hscale := iidTunedGapScale_pos T mu theta0 a sigmaSq
    hT hmu htheta ha hsigma
  rw [← sub_neg,
    tunedIidNoisyRisk_gap_exact T mu theta0 a sigmaSq
      hT hmu htheta ha hsigma.le]
  constructor
  · intro hneg
    by_contra hnot
    have hadv : iidCoherenceAdvantage T mu <= 0 := le_of_not_gt hnot
    have : 0 <= -(iidTunedGapScale T mu theta0 a sigmaSq *
        iidCoherenceAdvantage T mu) := by
      exact neg_nonneg.mpr (mul_nonpos_of_nonneg_of_nonpos hscale.le hadv)
    exact (not_lt_of_ge this) hneg
  · intro hadv
    exact neg_neg_of_pos (mul_pos hscale hadv)

/-- **Exactly retuned NEUTRAL criterion.** -/
theorem tunedMomentum_neutral_iff
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (htheta : theta0 ≠ 0)
    (ha : 0 < a) (hsigma : 0 < sigmaSq) :
    tunedIidNoisyRisk T mu theta0 a sigmaSq =
        tunedIidNoisyRisk T 0 theta0 a sigmaSq <->
      iidCoherenceAdvantage T mu = 0 := by
  have hscale := iidTunedGapScale_pos T mu theta0 a sigmaSq
    hT hmu htheta ha hsigma
  rw [← sub_eq_zero,
    tunedIidNoisyRisk_gap_exact T mu theta0 a sigmaSq
      hT hmu htheta ha hsigma.le]
  simp [hscale.ne']

/-- **Exactly retuned HURTS criterion.** -/
theorem tunedMomentum_hurts_iff
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (htheta : theta0 ≠ 0)
    (ha : 0 < a) (hsigma : 0 < sigmaSq) :
    tunedIidNoisyRisk T 0 theta0 a sigmaSq <
        tunedIidNoisyRisk T mu theta0 a sigmaSq <->
      iidCoherenceAdvantage T mu < 0 := by
  have hscale := iidTunedGapScale_pos T mu theta0 a sigmaSq
    hT hmu htheta ha hsigma
  rw [← sub_pos,
    tunedIidNoisyRisk_gap_exact T mu theta0 a sigmaSq
      hT hmu htheta ha hsigma.le]
  constructor
  · intro hpos
    by_contra hnot
    have hadv : 0 <= iidCoherenceAdvantage T mu := le_of_not_gt hnot
    have : -(iidTunedGapScale T mu theta0 a sigmaSq *
        iidCoherenceAdvantage T mu) <= 0 := by
      exact neg_nonpos.mpr (mul_nonneg hscale.le hadv)
    exact (not_lt_of_ge this) hpos
  · intro hadv
    exact neg_pos.mpr (mul_neg_of_pos_of_neg hscale hadv)

/-- At zero noise, both exactly retuned scalar risks are zero. -/
theorem tunedIidNoisyRisk_zero_noise
    (T : Nat) (mu theta0 a : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (htheta : theta0 ≠ 0) (ha : 0 < a) :
    tunedIidNoisyRisk T mu theta0 a 0 = 0 := by
  have hden := iidNoisyDenom_pos T mu theta0 a 0
    hT hmu htheta ha (by norm_num)
  unfold tunedIidNoisyRisk iidNoisyFrozenRisk iidNoisyOptimalEta
  rw [heavyBallSecondMomentRisk_at_optimal T mu theta0 a 0
    (iidOuterNoiseFactor T mu) hden.ne']
  simp

/-- One update is neutral: mean and iid noise receive the same multiplier. -/
theorem iidCoherenceAdvantage_one (mu : Real) :
    iidCoherenceAdvantage 1 mu = 0 := by
  norm_num [iidCoherenceAdvantage, iidOuterNoiseFactor, effectiveCoeff,
    terminalMultiplier, geometricPrefix, sum_range_succ]

/-- At two updates the exact coherence advantage is `mu * (4-mu)`. -/
theorem iidCoherenceAdvantage_two (mu : Real) :
    iidCoherenceAdvantage 2 mu = mu * (4 - mu) := by
  norm_num [iidCoherenceAdvantage, iidOuterNoiseFactor, effectiveCoeff,
    terminalMultiplier, geometricPrefix, sum_range_succ]
  ring

/-- Thus positive stable momentum strictly helps at `T = 2` in the exactly
retuned iid scalar surrogate. -/
theorem iidCoherenceAdvantage_two_pos
    (mu : Real) (hmu0 : 0 < mu) (hmu1 : mu < 1) :
    0 < iidCoherenceAdvantage 2 mu := by
  rw [iidCoherenceAdvantage_two]
  exact mul_pos hmu0 (by linarith)

/-- A positive-factor expansion of the `T = 5` iid coherence advantage. -/
theorem iidCoherenceAdvantage_five (mu : Real) :
    iidCoherenceAdvantage 5 mu =
      40 * mu + 26 * mu ^ 2 + 44 * mu ^ 3 + 20 * mu ^ 4 +
        20 * mu ^ 5 + 4 * mu ^ 7 * (1 - mu) := by
  norm_num [iidCoherenceAdvantage, iidOuterNoiseFactor, effectiveCoeff,
    terminalMultiplier, geometricPrefix, sum_range_succ]
  ring

/-- The exactly retuned iid scalar model predicts HELP at `T = 5` for every
strictly positive stable momentum. -/
theorem iidCoherenceAdvantage_five_pos
    (mu : Real) (hmu0 : 0 < mu) (hmu1 : mu < 1) :
    0 < iidCoherenceAdvantage 5 mu := by
  rw [iidCoherenceAdvantage_five]
  positivity

/-- Machine-checked statement of the `T = 5` qualitative mismatch: under iid
marginal noise and exact scalar retuning, stable positive momentum strictly
beats no momentum. -/
theorem tunedMomentum_helps_T5_iid
    (mu theta0 a sigmaSq : Real) (hmu0 : 0 < mu) (hmu1 : mu < 1)
    (htheta : theta0 ≠ 0) (ha : 0 < a) (hsigma : 0 < sigmaSq) :
    tunedIidNoisyRisk 5 mu theta0 a sigmaSq <
      tunedIidNoisyRisk 5 0 theta0 a sigmaSq := by
  exact (tunedMomentum_helps_iff 5 mu theta0 a sigmaSq
    (by norm_num) hmu0.le htheta ha hsigma).2
      (iidCoherenceAdvantage_five_pos mu hmu0 hmu1)

/-! ## Noise-scale boundary for the finite steady prescription -/

/-- Finite-horizon mean residual of the asymptotic `(1-mu)` prescription. -/
def steadyPrescriptionBias (T : Nat) (mu : Real) : Real :=
  1 - effectiveCoeff .heavyBall T mu / (T : Real) * (1 - mu)

/-- Iid variance saved by scaling the heavy-ball rate by `(1-mu)`, measured
relative to the no-momentum variance factor `T`. -/
def steadyPrescriptionVarianceSaving (T : Nat) (mu : Real) : Real :=
  (T : Real) - (1 - mu) ^ 2 * iidOuterNoiseFactor T mu

/-- Noisy risk of the asymptotic steady-state rate at a finite horizon. -/
def steadyPrescriptionIidRisk
    (T : Nat) (mu theta0 a sigmaSq : Real) : Real :=
  iidNoisyFrozenRisk T mu
    (steadyStatePrescribedEta .heavyBall T mu a) theta0 a sigmaSq

/-- Exact bias-versus-variance factorization for the finite prescription. -/
theorem steadyPrescriptionIidRisk_gap_exact
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (ha : 0 < a) :
    steadyPrescriptionIidRisk T mu theta0 a sigmaSq -
        steadyPrescriptionIidRisk T 0 theta0 a sigmaSq =
      ((a / 2) / (a ^ 2 * (T : Real) ^ 2)) *
        (a ^ 2 * (T : Real) ^ 2 * theta0 ^ 2 *
            steadyPrescriptionBias T mu ^ 2 -
          sigmaSq * steadyPrescriptionVarianceSaving T mu) := by
  have hTreal : (T : Real) ≠ 0 := by exact_mod_cast hT.ne'
  have hdetMu :
      frozenQuadraticJ .heavyBall T mu
          (steadyStatePrescribedEta .heavyBall T mu a) theta0 a =
        (a / 2) * theta0 ^ 2 * steadyPrescriptionBias T mu ^ 2 := by
    have htarget :
        frozenQuadraticJ .heavyBall T mu
          (alignedOptimalEta .heavyBall T mu a) theta0 a = 0 := by
      simp [frozenQuadraticJ, frozenGradientQuadraticLoss,
        alignedOptimalEta_reaches_zero .heavyBall T mu theta0 a hT hmu ha]
    have hexact := steadyStatePrescriptionPenalty_exact
      .heavyBall T mu theta0 a hT hmu ha
    unfold steadyStatePrescriptionPenalty at hexact
    rw [htarget, sub_zero] at hexact
    simpa [steadyPrescriptionBias] using hexact
  have hdetZero :
      frozenQuadraticJ .heavyBall T 0
          (steadyStatePrescribedEta .heavyBall T 0 a) theta0 a = 0 := by
    have hzero : steadyStatePrescribedEta .heavyBall T 0 a =
        alignedOptimalEta .heavyBall T 0 a := by
      simp [steadyStatePrescribedEta]
    rw [hzero]
    unfold frozenQuadraticJ frozenGradientQuadraticLoss
    rw [alignedOptimalEta_reaches_zero .heavyBall T 0 theta0 a hT
      (by norm_num) ha]
    norm_num
  unfold steadyPrescriptionIidRisk iidNoisyFrozenRisk
    heavyBallSecondMomentRisk
  rw [hdetMu, hdetZero]
  unfold steadyStatePrescribedEta alignedOptimalEta
    steadyPrescriptionVarianceSaving
  simp only [effectiveCoeff_zero_momentum, iidOuterNoiseFactor_zero_momentum]
  field_simp [hTreal, ha.ne']
  ring

/-- Each rate-scaled iid buffer variance is strictly below the no-momentum
unit variance when `0 < mu < 1`. -/
theorem scaledIidStepVariance_lt_one
    (t : Nat) (mu : Real) (hmu0 : 0 < mu) (hmu1 : mu < 1) :
    (1 - mu) ^ 2 * geometricPrefix (t + 1) (mu ^ 2) < 1 := by
  have hmu2 : mu ^ 2 < 1 := by nlinarith [sq_nonneg (1 - mu)]
  have hden : 0 < 1 - mu ^ 2 := sub_pos.mpr hmu2
  have hpow : 0 < (mu ^ 2) ^ (t + 1) := by positivity
  have hprefix : geometricPrefix (t + 1) (mu ^ 2) < 1 / (1 - mu ^ 2) := by
    rw [geometricPrefix_eq_div (t + 1) (mu ^ 2) (ne_of_lt hmu2)]
    apply (div_lt_div_iff_of_pos_right hden).2
    nlinarith
  calc
    (1 - mu) ^ 2 * geometricPrefix (t + 1) (mu ^ 2) <
        (1 - mu) ^ 2 * (1 / (1 - mu ^ 2)) := by
          exact mul_lt_mul_of_pos_left hprefix (sq_pos_of_pos (sub_pos.mpr hmu1))
    _ = (1 - mu) / (1 + mu) := by
      have hleft : 1 - mu ≠ 0 := (sub_pos.mpr hmu1).ne'
      have hright : 1 + mu ≠ 0 := by positivity
      field_simp [hleft, hright]
      ring
    _ < 1 := by
      exact (div_lt_one (by positivity : 0 < 1 + mu)).2 (by linarith)

/-- The finite steady prescription has a strictly positive iid variance saving
for every nonempty horizon and strictly positive stable momentum. -/
theorem steadyPrescriptionVarianceSaving_pos
    (T : Nat) (mu : Real) (hT : 0 < T) (hmu0 : 0 < mu) (hmu1 : mu < 1) :
    0 < steadyPrescriptionVarianceSaving T mu := by
  have hsum :
      (1 - mu) ^ 2 * iidOuterNoiseFactor T mu < (T : Real) := by
    unfold iidOuterNoiseFactor effectiveCoeff
    rw [mul_sum]
    have hall : ∀ t ∈ range T,
        (1 - mu) ^ 2 * terminalMultiplier .heavyBall (t + 1) (mu ^ 2) <= 1 := by
      intro t _ht
      exact (scaledIidStepVariance_lt_one t mu hmu0 hmu1).le
    have hexists : ∃ t ∈ range T,
        (1 - mu) ^ 2 * terminalMultiplier .heavyBall (t + 1) (mu ^ 2) < 1 := by
      refine ⟨0, mem_range.mpr hT, ?_⟩
      exact scaledIidStepVariance_lt_one 0 mu hmu0 hmu1
    have := sum_lt_sum hall hexists
    simpa only [sum_const, card_range, nsmul_eq_mul, mul_one] using this
  unfold steadyPrescriptionVarianceSaving
  linarith

/-- Every finite heavy-ball mean multiplier is strictly below its steady
value after normalization by `(1-mu)`. -/
theorem scaledMeanStep_lt_one
    (t : Nat) (mu : Real) (hmu0 : 0 < mu) (hmu1 : mu < 1) :
    (1 - mu) * geometricPrefix (t + 1) mu < 1 := by
  have hden : 1 - mu ≠ 0 := (sub_pos.mpr hmu1).ne'
  have heq :
      (1 - mu) * ((1 - mu ^ (t + 1)) / (1 - mu)) =
        1 - mu ^ (t + 1) := by
    field_simp [hden]
  rw [geometricPrefix_eq_div (t + 1) mu (ne_of_lt hmu1), heq]
  have hpow : 0 < mu ^ (t + 1) := by positivity
  linarith

/-- Consequently the asymptotic steady prescription has a strictly positive
deterministic residual at every finite nonempty horizon. -/
theorem steadyPrescriptionBias_pos
    (T : Nat) (mu : Real) (hT : 0 < T) (hmu0 : 0 < mu) (hmu1 : mu < 1) :
    0 < steadyPrescriptionBias T mu := by
  have hsum :
      (1 - mu) * effectiveCoeff .heavyBall T mu < (T : Real) := by
    unfold effectiveCoeff
    rw [mul_sum]
    have hall : ∀ t ∈ range T,
        (1 - mu) * terminalMultiplier .heavyBall (t + 1) mu <= 1 := by
      intro t _ht
      exact (scaledMeanStep_lt_one t mu hmu0 hmu1).le
    have hexists : ∃ t ∈ range T,
        (1 - mu) * terminalMultiplier .heavyBall (t + 1) mu < 1 := by
      refine ⟨0, mem_range.mpr hT, ?_⟩
      exact scaledMeanStep_lt_one 0 mu hmu0 hmu1
    have := sum_lt_sum hall hexists
    simpa only [sum_const, card_range, nsmul_eq_mul, mul_one] using this
  have hTreal : (0 : Real) < T := by exact_mod_cast hT
  have hnormalized :
      effectiveCoeff .heavyBall T mu / (T : Real) * (1 - mu) < 1 := by
    rw [show effectiveCoeff .heavyBall T mu / (T : Real) * (1 - mu) =
      ((1 - mu) * effectiveCoeff .heavyBall T mu) / (T : Real) by ring]
    exact (div_lt_one hTreal).2 hsum
  unfold steadyPrescriptionBias
  linarith

/-- **Finite-prescription HELP criterion.** -/
theorem steadyPrescription_helps_iff
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (ha : 0 < a) :
    steadyPrescriptionIidRisk T mu theta0 a sigmaSq <
        steadyPrescriptionIidRisk T 0 theta0 a sigmaSq <->
      a ^ 2 * (T : Real) ^ 2 * theta0 ^ 2 * steadyPrescriptionBias T mu ^ 2 <
        sigmaSq * steadyPrescriptionVarianceSaving T mu := by
  have hscale : 0 < (a / 2) / (a ^ 2 * (T : Real) ^ 2) := by
    have hTreal : (0 : Real) < T := by exact_mod_cast hT
    positivity
  rw [← sub_neg,
    steadyPrescriptionIidRisk_gap_exact T mu theta0 a sigmaSq hT hmu ha]
  constructor
  · intro hneg
    by_contra hnot
    have hinside : 0 <=
        a ^ 2 * (T : Real) ^ 2 * theta0 ^ 2 * steadyPrescriptionBias T mu ^ 2 -
          sigmaSq * steadyPrescriptionVarianceSaving T mu :=
      sub_nonneg.mpr (le_of_not_gt hnot)
    exact (not_lt_of_ge (mul_nonneg hscale.le hinside)) hneg
  · intro hlt
    exact mul_neg_of_pos_of_neg hscale (sub_neg.mpr hlt)

/-- **Finite-prescription NEUTRAL criterion.** -/
theorem steadyPrescription_neutral_iff
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (ha : 0 < a) :
    steadyPrescriptionIidRisk T mu theta0 a sigmaSq =
        steadyPrescriptionIidRisk T 0 theta0 a sigmaSq <->
      a ^ 2 * (T : Real) ^ 2 * theta0 ^ 2 * steadyPrescriptionBias T mu ^ 2 =
        sigmaSq * steadyPrescriptionVarianceSaving T mu := by
  have hscale : (a / 2) / (a ^ 2 * (T : Real) ^ 2) ≠ 0 := by
    have hTreal : (T : Real) ≠ 0 := by exact_mod_cast hT.ne'
    positivity
  rw [← sub_eq_zero,
    steadyPrescriptionIidRisk_gap_exact T mu theta0 a sigmaSq hT hmu ha]
  constructor
  · intro hzero
    exact sub_eq_zero.mp ((mul_eq_zero.mp hzero).resolve_left hscale)
  · intro heq
    rw [sub_eq_zero.mpr heq, mul_zero]

/-- **Finite-prescription HURTS criterion.** -/
theorem steadyPrescription_hurts_iff
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (ha : 0 < a) :
    steadyPrescriptionIidRisk T 0 theta0 a sigmaSq <
        steadyPrescriptionIidRisk T mu theta0 a sigmaSq <->
      sigmaSq * steadyPrescriptionVarianceSaving T mu <
        a ^ 2 * (T : Real) ^ 2 * theta0 ^ 2 * steadyPrescriptionBias T mu ^ 2 := by
  have hscale : 0 < (a / 2) / (a ^ 2 * (T : Real) ^ 2) := by
    have hTreal : (0 : Real) < T := by exact_mod_cast hT
    positivity
  rw [← sub_pos,
    steadyPrescriptionIidRisk_gap_exact T mu theta0 a sigmaSq hT hmu ha]
  constructor
  · intro hpos
    by_contra hnot
    have hinside :
        a ^ 2 * (T : Real) ^ 2 * theta0 ^ 2 * steadyPrescriptionBias T mu ^ 2 -
            sigmaSq * steadyPrescriptionVarianceSaving T mu <= 0 :=
      sub_nonpos.mpr (le_of_not_gt hnot)
    exact (not_lt_of_ge (mul_nonpos_of_nonneg_of_nonpos hscale.le hinside)) hpos
  · intro hlt
    exact mul_pos hscale (sub_pos.mpr hlt)

/-- Dimensionless critical noise-to-signal ratio for the finite prescription. -/
def steadyPrescriptionCriticalNoiseRatio (T : Nat) (mu : Real) : Real :=
  (T : Real) ^ 2 * steadyPrescriptionBias T mu ^ 2 /
    steadyPrescriptionVarianceSaving T mu

/-- Normalized form of the finite-prescription boundary. -/
theorem steadyPrescription_helps_iff_noiseRatio
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu0 : 0 < mu) (hmu1 : mu < 1)
    (htheta : theta0 ≠ 0) (ha : 0 < a) :
    steadyPrescriptionIidRisk T mu theta0 a sigmaSq <
        steadyPrescriptionIidRisk T 0 theta0 a sigmaSq <->
      steadyPrescriptionCriticalNoiseRatio T mu <
        sigmaSq / (a ^ 2 * theta0 ^ 2) := by
  have hsave := steadyPrescriptionVarianceSaving_pos T mu hT hmu0 hmu1
  have hsignal : 0 < a ^ 2 * theta0 ^ 2 := by positivity
  rw [steadyPrescription_helps_iff T mu theta0 a sigmaSq hT hmu0.le ha]
  unfold steadyPrescriptionCriticalNoiseRatio
  rw [div_lt_div_iff₀ hsave hsignal]
  ring_nf

/-- Normalized HURTS side of the same finite-prescription boundary. -/
theorem steadyPrescription_hurts_iff_noiseRatio
    (T : Nat) (mu theta0 a sigmaSq : Real)
    (hT : 0 < T) (hmu0 : 0 < mu) (hmu1 : mu < 1)
    (htheta : theta0 ≠ 0) (ha : 0 < a) :
    steadyPrescriptionIidRisk T 0 theta0 a sigmaSq <
        steadyPrescriptionIidRisk T mu theta0 a sigmaSq <->
      sigmaSq / (a ^ 2 * theta0 ^ 2) <
        steadyPrescriptionCriticalNoiseRatio T mu := by
  have hsave := steadyPrescriptionVarianceSaving_pos T mu hT hmu0 hmu1
  have hsignal : 0 < a ^ 2 * theta0 ^ 2 := by positivity
  rw [steadyPrescription_hurts_iff T mu theta0 a sigmaSq hT hmu0.le ha]
  unfold steadyPrescriptionCriticalNoiseRatio
  rw [div_lt_div_iff₀ hsignal hsave]
  ring_nf

/-- In particular, zero per-step noise lies strictly on the HURTS side of the
finite steady-prescription boundary. -/
theorem steadyPrescription_hurts_at_zero_noise
    (T : Nat) (mu theta0 a : Real)
    (hT : 0 < T) (hmu0 : 0 < mu) (hmu1 : mu < 1)
    (htheta : theta0 ≠ 0) (ha : 0 < a) :
    steadyPrescriptionIidRisk T 0 theta0 a 0 <
      steadyPrescriptionIidRisk T mu theta0 a 0 := by
  apply (steadyPrescription_hurts_iff T mu theta0 a 0 hT hmu0.le ha).2
  have hbias := steadyPrescriptionBias_pos T mu hT hmu0 hmu1
  have hTreal : (0 : Real) < T := by exact_mod_cast hT
  rw [zero_mul]
  have hpositive :
      0 < a ^ 2 * (T : Real) ^ 2 * theta0 ^ 2 *
        steadyPrescriptionBias T mu ^ 2 := by
    positivity
  exact hpositive

/-- Paper-facing `T = 5` HURTS boundary for the finite steady prescription. -/
theorem snooHurts_T5_criterion
    (mu theta0 a sigmaSq : Real) (hmu : 0 <= mu) (ha : 0 < a) :
    steadyPrescriptionIidRisk 5 0 theta0 a sigmaSq <
        steadyPrescriptionIidRisk 5 mu theta0 a sigmaSq <->
      sigmaSq * steadyPrescriptionVarianceSaving 5 mu <
        a ^ 2 * 25 * theta0 ^ 2 * steadyPrescriptionBias 5 mu ^ 2 := by
  convert steadyPrescription_hurts_iff 5 mu theta0 a sigmaSq
    (by norm_num) hmu ha using 1
  all_goals norm_num

end

end LeanMechanism
