import Mathlib

/-!
# Finite-age multipliers for zero-started FedAdam

This file is the scalar, constant-pseudo-gradient algebra behind the registered
FedAdam finite-age prediction.  Every statement is coordinatewise: a vector
version follows by applying it to each coordinate.

Reddi et al. (2021) update the server moments without Adam bias correction,

`m_t = beta1 * m_(t-1) + (1-beta1) * g`,
`v_t = beta2 * v_(t-1) + (1-beta2) * g^2`,

and apply `m_t / (sqrt v_t + epsilon)`.  Here both moments start at zero and
the first actual server call has age one.  For a constant `g`, the two startup
shapes are therefore `1-beta1^t` and `1-beta2^t`; the latter enters the applied
direction through a square root.

`rawTunedDeviation` is the inverse finite-age gain relative to the steady-age
gain.  It is the terminal-step heuristic registered by the experiment, not the
sum of all preceding step multipliers.  The latter is kept separately as
`rawEffectiveCoeff`, exactly as `FiniteHorizonOuter.lean` separates terminal
and accumulated coefficients.

The final section also separates two notions often both called "Adam bias
correction":

* corrected moments (the PyTorch mathematical convention) put epsilon after
  `sqrt (v_t / (1-beta2^t))` and are exactly age-invariant here;
* the original-Adam scalar-prefactor form multiplies the raw-denominator step
  by `sqrt (1-beta2^t) / (1-beta1^t)`.  It agrees when epsilon is zero, but a
  positive raw-denominator epsilon leaves a residual transient.
-/

namespace LeanMechanism

noncomputable section

open Finset
open scoped BigOperators

/-! ## Zero-start moment recurrences -/

/-- Zero-started first moment under a constant scalar pseudo-gradient. -/
def adamFirstMoment (beta g : Real) : Nat -> Real
  | 0 => 0
  | t + 1 => beta * adamFirstMoment beta g t + (1 - beta) * g

/-- Zero-started second moment under a constant scalar pseudo-gradient. -/
def adamSecondMoment (beta g : Real) : Nat -> Real
  | 0 => 0
  | t + 1 => beta * adamSecondMoment beta g t + (1 - beta) * g ^ 2

@[simp]
theorem adamFirstMoment_zero (beta g : Real) : adamFirstMoment beta g 0 = 0 := rfl

@[simp]
theorem adamFirstMoment_succ (beta g : Real) (t : Nat) :
    adamFirstMoment beta g (t + 1) =
      beta * adamFirstMoment beta g t + (1 - beta) * g := rfl

@[simp]
theorem adamSecondMoment_zero (beta g : Real) : adamSecondMoment beta g 0 = 0 := rfl

@[simp]
theorem adamSecondMoment_succ (beta g : Real) (t : Nat) :
    adamSecondMoment beta g (t + 1) =
      beta * adamSecondMoment beta g t + (1 - beta) * g ^ 2 := rfl

/-- The common zero-start EMA startup shape. -/
def adamStartupShape (beta : Real) (t : Nat) : Real :=
  1 - beta ^ t

/-- The second-moment startup shape after the denominator square root. -/
def adamRootStartupShape (beta : Real) (t : Nat) : Real :=
  Real.sqrt (adamStartupShape beta t)

/-- Exact first-moment unrolling from `m_0 = 0`. -/
theorem adamFirstMoment_closed_form (beta g : Real) (t : Nat) :
    adamFirstMoment beta g t = adamStartupShape beta t * g := by
  induction t with
  | zero => simp [adamStartupShape]
  | succ t ih =>
      rw [adamFirstMoment_succ, ih]
      simp only [adamStartupShape, pow_succ]
      ring

/-- Exact second-moment unrolling from `v_0 = 0`. -/
theorem adamSecondMoment_closed_form (beta g : Real) (t : Nat) :
    adamSecondMoment beta g t = adamStartupShape beta t * g ^ 2 := by
  induction t with
  | zero => simp [adamStartupShape]
  | succ t ih =>
      rw [adamSecondMoment_succ, ih]
      simp only [adamStartupShape, pow_succ]
      ring

theorem adamStartupShape_nonneg
    (beta : Real) (t : Nat) (hbeta0 : 0 <= beta) (hbeta1 : beta <= 1) :
    0 <= adamStartupShape beta t := by
  exact sub_nonneg.mpr (pow_le_one₀ hbeta0 hbeta1)

theorem adamStartupShape_pos
    (beta : Real) (t : Nat) (hbeta0 : 0 <= beta) (hbeta1 : beta < 1)
    (ht : 0 < t) :
    0 < adamStartupShape beta t := by
  exact sub_pos.mpr (pow_lt_one₀ hbeta0 hbeta1 ht.ne')

theorem adamRootStartupShape_nonneg (beta : Real) (t : Nat) :
    0 <= adamRootStartupShape beta t := by
  exact Real.sqrt_nonneg _

theorem adamRootStartupShape_pos
    (beta : Real) (t : Nat) (hbeta0 : 0 <= beta) (hbeta1 : beta < 1)
    (ht : 0 < t) :
    0 < adamRootStartupShape beta t := by
  exact Real.sqrt_pos.2 (adamStartupShape_pos beta t hbeta0 hbeta1 ht)

/-- The square root of the zero-started second moment contains the square root
of the second-moment startup shape, times `|g|`. -/
theorem sqrt_adamSecondMoment_closed_form
    (beta g : Real) (t : Nat) (hbeta0 : 0 <= beta) (hbeta1 : beta <= 1) :
    Real.sqrt (adamSecondMoment beta g t) =
      adamRootStartupShape beta t * |g| := by
  rw [adamSecondMoment_closed_form, adamRootStartupShape]
  rw [Real.sqrt_mul (adamStartupShape_nonneg beta t hbeta0 hbeta1)]
  rw [Real.sqrt_sq_eq_abs]

/-! ## Reddi-style raw FedAdam and the registered deviation -/

/-- Reddi-style (uncorrected) scalar FedAdam direction on call `t`. -/
def rawFedAdamDirection
    (beta1 beta2 epsilon g : Real) (t : Nat) : Real :=
  adamFirstMoment beta1 g t /
    (Real.sqrt (adamSecondMoment beta2 g t) + epsilon)

/-- Dimensionless direction multiplier for a positive constant coordinate,
where `q = epsilon / g`. -/
def rawFedAdamMultiplier
    (beta1 beta2 q : Real) (t : Nat) : Real :=
  adamStartupShape beta1 t / (adamRootStartupShape beta2 t + q)

/-- The code-level raw direction reduces to the dimensionless multiplier after
writing the floor as `q = epsilon / g`. -/
theorem rawFedAdamDirection_closed_form
    (beta1 beta2 epsilon g : Real) (t : Nat)
    (hbeta20 : 0 <= beta2) (hbeta21 : beta2 <= 1) (hg : 0 < g) :
    rawFedAdamDirection beta1 beta2 epsilon g t =
      rawFedAdamMultiplier beta1 beta2 (epsilon / g) t := by
  rw [rawFedAdamDirection, adamFirstMoment_closed_form,
    sqrt_adamSecondMoment_closed_form beta2 g t hbeta20 hbeta21,
    abs_of_pos hg]
  unfold rawFedAdamMultiplier
  have hdenom :
      adamRootStartupShape beta2 t * g + epsilon =
        g * (adamRootStartupShape beta2 t + epsilon / g) := by
    field_simp [hg.ne']
  rw [hdenom]
  field_simp [hg.ne']

/-- Steady-age multiplier for a constant positive coordinate. -/
def rawFedAdamSteadyMultiplier (q : Real) : Real :=
  1 / (1 + q)

/-- Finite-age gain divided by its steady-age value. -/
def rawNormalizedAgeGain
    (beta1 beta2 q : Real) (t : Nat) : Real :=
  adamStartupShape beta1 t * (1 + q) /
    (adamRootStartupShape beta2 t + q)

/-- Registered tuned-rate deviation: the inverse normalized terminal gain.

For `q = 0`, this is
`sqrt (1-beta2^t) / (1-beta1^t)`, not a Nesterov geometric deficit.
-/
def rawTunedDeviation
    (beta1 beta2 q : Real) (t : Nat) : Real :=
  (adamRootStartupShape beta2 t + q) /
    (adamStartupShape beta1 t * (1 + q))

/-- Gain and tuned-rate deviation are exact reciprocals whenever the displayed
factors are nonzero. -/
theorem rawTunedDeviation_mul_normalizedAgeGain
    (beta1 beta2 q : Real) (t : Nat)
    (hfirst : adamStartupShape beta1 t ≠ 0)
    (hfloor : 1 + q ≠ 0)
    (hroot : adamRootStartupShape beta2 t + q ≠ 0) :
    rawTunedDeviation beta1 beta2 q t *
        rawNormalizedAgeGain beta1 beta2 q t = 1 := by
  unfold rawTunedDeviation rawNormalizedAgeGain
  field_simp

@[simp]
theorem rawTunedDeviation_no_floor (beta1 beta2 : Real) (t : Nat) :
    rawTunedDeviation beta1 beta2 0 t =
      adamRootStartupShape beta2 t / adamStartupShape beta1 t := by
  simp [rawTunedDeviation]

/-- Squaring the no-floor prediction exposes a purely rational expression;
this is convenient for exact registered-point comparisons. -/
theorem rawTunedDeviation_no_floor_sq
    (beta1 beta2 : Real) (t : Nat)
    (hbeta20 : 0 <= beta2) (hbeta21 : beta2 <= 1) :
    (rawTunedDeviation beta1 beta2 0 t) ^ 2 =
      adamStartupShape beta2 t / (adamStartupShape beta1 t) ^ 2 := by
  rw [rawTunedDeviation_no_floor, div_pow]
  exact congrArg (fun x => x / adamStartupShape beta1 t ^ 2)
    (Real.sq_sqrt (adamStartupShape_nonneg beta2 t hbeta20 hbeta21))

/-- Exact algebra for the epsilon-floor shift.  At a fixed finite age it raises
the deviation whenever `q >= 0` and the root startup shape is at most one. -/
theorem rawTunedDeviation_floor_difference
    (beta1 beta2 q : Real) (t : Nat)
    (hfirst : adamStartupShape beta1 t ≠ 0) (hfloor : 1 + q ≠ 0) :
    rawTunedDeviation beta1 beta2 q t -
        rawTunedDeviation beta1 beta2 0 t =
      q * (1 - adamRootStartupShape beta2 t) /
        (adamStartupShape beta1 t * (1 + q)) := by
  unfold rawTunedDeviation
  field_simp
  ring

theorem adamRootStartupShape_le_one
    (beta : Real) (t : Nat) (hbeta0 : 0 <= beta) :
    adamRootStartupShape beta t <= 1 := by
  apply (Real.sqrt_le_iff).2
  refine ⟨by norm_num, ?_⟩
  unfold adamStartupShape
  have hpow : 0 <= beta ^ t := pow_nonneg hbeta0 t
  nlinarith

/-- A nonnegative epsilon floor can only increase the registered deviation at
a fixed positive age. -/
theorem rawTunedDeviation_no_floor_le_floor
    (beta1 beta2 q : Real) (t : Nat)
    (hbeta10 : 0 <= beta1) (hbeta11 : beta1 < 1)
    (hbeta20 : 0 <= beta2)
    (hq : 0 <= q) (ht : 0 < t) :
    rawTunedDeviation beta1 beta2 0 t <=
      rawTunedDeviation beta1 beta2 q t := by
  have hfirstpos := adamStartupShape_pos beta1 t hbeta10 hbeta11 ht
  have hfloorpos : 0 < 1 + q := by linarith
  rw [sub_nonneg.symm,
    rawTunedDeviation_floor_difference beta1 beta2 q t
      hfirstpos.ne' hfloorpos.ne']
  positivity [adamRootStartupShape_le_one beta2 t hbeta20]

/-! ## Terminal multiplier versus accumulated coefficient -/

/-- Sum of the actual call multipliers at ages `1, ..., T`. -/
def rawEffectiveCoeff
    (beta1 beta2 q : Real) (T : Nat) : Real :=
  ∑ t ∈ range T, rawFedAdamMultiplier beta1 beta2 q (t + 1)

/-- Total scalar displacement in the normalized positive-coordinate model. -/
def rawAccumulatedDisplacement
    (beta1 beta2 q eta : Real) (T : Nat) : Real :=
  eta * ∑ t ∈ range T, rawFedAdamMultiplier beta1 beta2 q (t + 1)

theorem rawAccumulatedDisplacement_closed_form
    (beta1 beta2 q eta : Real) (T : Nat) :
    rawAccumulatedDisplacement beta1 beta2 q eta T =
      eta * rawEffectiveCoeff beta1 beta2 q T := by
  rfl

/-! ## Bias-correction conventions -/

/-- Correct-the-moments convention: both moment startup factors are divided
out before epsilon is added to the corrected RMS. -/
def pytorchCorrectedMultiplier
    (beta1 beta2 q : Real) (t : Nat) : Real :=
  (adamStartupShape beta1 t / adamStartupShape beta1 t) /
    (Real.sqrt
      (adamStartupShape beta2 t / adamStartupShape beta2 t) + q)

/-- Full corrected-moment Adam is exactly age-invariant on a constant input,
including with a positive epsilon floor. -/
theorem pytorchCorrectedMultiplier_eq_steady
    (beta1 beta2 q : Real) (t : Nat)
    (hfirst : adamStartupShape beta1 t ≠ 0)
    (hsecond : adamStartupShape beta2 t ≠ 0) :
    pytorchCorrectedMultiplier beta1 beta2 q t =
      rawFedAdamSteadyMultiplier q := by
  simp [pytorchCorrectedMultiplier, rawFedAdamSteadyMultiplier, hfirst, hsecond]

/-- Original-Adam Algorithm-1 scalar-prefactor convention: multiply the raw
denominator direction by `sqrt(1-beta2^t)/(1-beta1^t)`. -/
def scalarPrefactorCorrectedMultiplier
    (beta1 beta2 q : Real) (t : Nat) : Real :=
  adamRootStartupShape beta2 t / adamStartupShape beta1 t *
    rawFedAdamMultiplier beta1 beta2 q t

theorem scalarPrefactorCorrectedMultiplier_closed_form
    (beta1 beta2 q : Real) (t : Nat)
    (hfirst : adamStartupShape beta1 t ≠ 0) :
    scalarPrefactorCorrectedMultiplier beta1 beta2 q t =
      adamRootStartupShape beta2 t /
        (adamRootStartupShape beta2 t + q) := by
  unfold scalarPrefactorCorrectedMultiplier rawFedAdamMultiplier
  field_simp

/-- Tuned-rate deviation left by the scalar-prefactor convention when epsilon
remains inside the raw (uncorrected) denominator. -/
def scalarPrefactorTunedDeviation
    (beta2 q : Real) (t : Nat) : Real :=
  (adamRootStartupShape beta2 t + q) /
    (adamRootStartupShape beta2 t * (1 + q))

@[simp]
theorem scalarPrefactorTunedDeviation_no_floor
    (beta2 : Real) (t : Nat)
    (hroot : adamRootStartupShape beta2 t ≠ 0) :
    scalarPrefactorTunedDeviation beta2 0 t = 1 := by
  simp [scalarPrefactorTunedDeviation, hroot]

/-- Exact residual caused by leaving epsilon in the raw denominator. -/
theorem scalarPrefactorTunedDeviation_sub_one
    (beta2 q : Real) (t : Nat)
    (hroot : adamRootStartupShape beta2 t ≠ 0) (hfloor : 1 + q ≠ 0) :
    scalarPrefactorTunedDeviation beta2 q t - 1 =
      q * (1 - adamRootStartupShape beta2 t) /
        (adamRootStartupShape beta2 t * (1 + q)) := by
  unfold scalarPrefactorTunedDeviation
  field_simp
  ring

/-! ## Exact registered no-floor shape at beta1=.9, beta2=.99 -/

/-- Square of the no-floor registered curve. -/
def registeredRawDeviationSq (t : Nat) : Real :=
  adamStartupShape (99 / 100 : Real) t /
    (adamStartupShape (9 / 10 : Real) t) ^ 2

/-- The four registered ages form a U-shape: `D(2) > D(5) > D(20)` but
`D(40) > D(20)`.  All four remain below the steady-age value one.  Since the
deviations are nonnegative, these exact square comparisons have the same order
as the deviations themselves. -/
theorem registeredRawDeviationSq_u_shape :
    registeredRawDeviationSq 5 < registeredRawDeviationSq 2 /\
    registeredRawDeviationSq 20 < registeredRawDeviationSq 5 /\
    registeredRawDeviationSq 20 < registeredRawDeviationSq 40 /\
    registeredRawDeviationSq 2 < 1 /\
    registeredRawDeviationSq 5 < 1 /\
    registeredRawDeviationSq 20 < 1 /\
    registeredRawDeviationSq 40 < 1 := by
  norm_num [registeredRawDeviationSq, adamStartupShape]

/-- The actual (unsquared) no-floor curve at the registered beta values. -/
def registeredRawDeviation (t : Nat) : Real :=
  rawTunedDeviation (9 / 10 : Real) (99 / 100 : Real) 0 t

theorem registeredRawDeviation_nonneg (t : Nat) (ht : 0 < t) :
    0 <= registeredRawDeviation t := by
  rw [registeredRawDeviation, rawTunedDeviation_no_floor]
  have hden : 0 < adamStartupShape (9 / 10 : Real) t :=
    adamStartupShape_pos (9 / 10 : Real) t (by norm_num) (by norm_num) ht
  exact div_nonneg (adamRootStartupShape_nonneg _ _) hden.le

theorem registeredRawDeviation_sq (t : Nat) :
    (registeredRawDeviation t) ^ 2 = registeredRawDeviationSq t := by
  rw [registeredRawDeviation, registeredRawDeviationSq,
    rawTunedDeviation_no_floor_sq (9 / 10 : Real) (99 / 100 : Real) t
      (by norm_num) (by norm_num)]

/-- Unsquared form of the exact registered U-shape. -/
theorem registeredRawDeviation_u_shape :
    registeredRawDeviation 5 < registeredRawDeviation 2 /\
    registeredRawDeviation 20 < registeredRawDeviation 5 /\
    registeredRawDeviation 20 < registeredRawDeviation 40 := by
  rcases registeredRawDeviationSq_u_shape with ⟨h25, h520, h2040, _⟩
  constructor
  · apply (sq_lt_sq₀ (registeredRawDeviation_nonneg 5 (by norm_num))
      (registeredRawDeviation_nonneg 2 (by norm_num))).1
    simpa only [registeredRawDeviation_sq] using h25
  constructor
  · apply (sq_lt_sq₀ (registeredRawDeviation_nonneg 20 (by norm_num))
      (registeredRawDeviation_nonneg 5 (by norm_num))).1
    simpa only [registeredRawDeviation_sq] using h520
  · apply (sq_lt_sq₀ (registeredRawDeviation_nonneg 20 (by norm_num))
      (registeredRawDeviation_nonneg 40 (by norm_num))).1
    simpa only [registeredRawDeviation_sq] using h2040

end

end LeanMechanism
