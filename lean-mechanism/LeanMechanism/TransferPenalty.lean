import LeanMechanism.QuadraticAlignment

/-!
# Finite-horizon transfer penalty

This file gives a closed transfer calculation for the scalar quadratic
`L(theta) = (a / 2) * theta^2` in the frozen-gradient model from
`QuadraticAlignment`.  If a learning rate is aligned at horizon `T1` and then
used at horizon `T2`, its terminal-loss excess is exactly

`(a / 2) * theta0^2 * (1 - c(T2,mu) / c(T1,mu))^2`,

where `c = effectiveCoeff`.  Hence the excess is nonnegative, and for nonzero
initial state it vanishes exactly when the accumulated multipliers match.

The exact excess is monotone in the relative multiplier mismatch
`|1 - c(T2,mu) / c(T1,mu)|`.  It is deliberately not claimed to be a function
of `|log(c(T1,mu) / c(T2,mu))|` alone: swapping `T1` and `T2` preserves that
absolute log-ratio but generally changes the directional terminal loss.  The
closed square formula is the stronger honest statement for the actual loss.

The all-horizon statements below inherit the scope of `frozenQuadraticJ`: the
gradient is frozen at `a * theta0`.  They are not claims about the exact
evolving-gradient recursion `quadraticJ`, whose all-horizon alignment law is
false.  The exact recursion agrees with this model at one update only.

Finally, the steady prescription `(1-mu) * eta0(T)` has vanishing same-horizon
penalty as `T -> infinity`.  Tuned transfer from `T` to `T+d` also vanishes for
each fixed offset `d`.  Merely requiring two unrelated horizons to diverge is
not enough: their accumulated-multiplier ratio must tend to one.
-/

namespace LeanMechanism

noncomputable section

open Filter
open scoped Topology

/-- Directional accumulated-multiplier ratio seen when a rate tuned at `T1`
is evaluated at `T2`. -/
def transferMultiplierRatio
    (rule : OuterRule) (T1 T2 : Nat) (mu : Real) : Real :=
  effectiveCoeff rule T2 mu / effectiveCoeff rule T1 mu

/-- Absolute relative accumulated-multiplier mismatch. -/
def relativeMultiplierMismatch
    (rule : OuterRule) (T1 T2 : Nat) (mu : Real) : Real :=
  |1 - transferMultiplierRatio rule T1 T2 mu|

/-- Terminal-loss excess from using the frozen-quadratic optimum tuned at
`T1` when the evaluation horizon is `T2`. -/
def tunedTransferPenalty
    (rule : OuterRule) (T1 T2 : Nat) (mu theta0 a : Real) : Real :=
  frozenQuadraticJ rule T2 mu (alignedOptimalEta rule T1 mu a) theta0 a -
    frozenQuadraticJ rule T2 mu (alignedOptimalEta rule T2 mu a) theta0 a

/-- Exact directional transfer-excess formula in the scalar frozen-gradient
quadratic. -/
theorem tunedTransferPenalty_exact
    (rule : OuterRule) (T1 T2 : Nat) (mu theta0 a : Real)
    (hT1 : 0 < T1) (hT2 : 0 < T2) (hmu : 0 <= mu) (ha : 0 < a) :
    tunedTransferPenalty rule T1 T2 mu theta0 a =
      (a / 2) * theta0 ^ 2 *
        (1 - transferMultiplierRatio rule T1 T2 mu) ^ 2 := by
  have hc1 : effectiveCoeff rule T1 mu ≠ 0 :=
    (effectiveCoeff_pos rule T1 mu hT1 hmu).ne'
  have htarget :
      frozenQuadraticJ rule T2 mu (alignedOptimalEta rule T2 mu a) theta0 a = 0 := by
    simp [frozenQuadraticJ, frozenGradientQuadraticLoss,
      alignedOptimalEta_reaches_zero rule T2 mu theta0 a hT2 hmu ha]
  rw [tunedTransferPenalty, htarget, sub_zero]
  unfold frozenQuadraticJ frozenGradientQuadraticLoss alignedOptimalEta
    transferMultiplierRatio
  field_simp [hc1, ha.ne']

/-- Transfer excess is a nonnegative loss gap. -/
theorem tunedTransferPenalty_nonneg
    (rule : OuterRule) (T1 T2 : Nat) (mu theta0 a : Real)
    (hT1 : 0 < T1) (hT2 : 0 < T2) (hmu : 0 <= mu) (ha : 0 < a) :
    0 <= tunedTransferPenalty rule T1 T2 mu theta0 a := by
  rw [tunedTransferPenalty_exact rule T1 T2 mu theta0 a hT1 hT2 hmu ha]
  positivity

/-- For positive curvature and nonzero initial state, zero transfer penalty is
equivalent to equality of the two accumulated multipliers. -/
theorem tunedTransferPenalty_eq_zero_iff
    (rule : OuterRule) (T1 T2 : Nat) (mu theta0 a : Real)
    (hT1 : 0 < T1) (hT2 : 0 < T2) (hmu : 0 <= mu)
    (ha : 0 < a) (htheta : theta0 ≠ 0) :
    tunedTransferPenalty rule T1 T2 mu theta0 a = 0 <->
      effectiveCoeff rule T1 mu = effectiveCoeff rule T2 mu := by
  have hc1 : effectiveCoeff rule T1 mu ≠ 0 :=
    (effectiveCoeff_pos rule T1 mu hT1 hmu).ne'
  rw [tunedTransferPenalty_exact rule T1 T2 mu theta0 a hT1 hT2 hmu ha]
  constructor
  · intro hzero
    have hscale : (a / 2) * theta0 ^ 2 ≠ 0 :=
      mul_ne_zero (div_ne_zero ha.ne' (by norm_num)) (pow_ne_zero 2 htheta)
    have hsquare :
        (1 - transferMultiplierRatio rule T1 T2 mu) ^ 2 = 0 :=
      (mul_eq_zero.mp hzero).resolve_left hscale
    have hratio : transferMultiplierRatio rule T1 T2 mu = 1 := by
      nlinarith [sq_nonneg (1 - transferMultiplierRatio rule T1 T2 mu)]
    unfold transferMultiplierRatio at hratio
    exact (div_eq_one_iff_eq hc1).mp hratio |>.symm
  · intro heq
    unfold transferMultiplierRatio
    rw [heq, div_self (effectiveCoeff_pos rule T2 mu hT2 hmu).ne']
    ring

/-- The exact loss is a positive scale times the square of the absolute
relative multiplier mismatch. -/
theorem tunedTransferPenalty_eq_mismatch_sq
    (rule : OuterRule) (T1 T2 : Nat) (mu theta0 a : Real)
    (hT1 : 0 < T1) (hT2 : 0 < T2) (hmu : 0 <= mu) (ha : 0 < a) :
    tunedTransferPenalty rule T1 T2 mu theta0 a =
      (a / 2) * theta0 ^ 2 *
        relativeMultiplierMismatch rule T1 T2 mu ^ 2 := by
  rw [tunedTransferPenalty_exact rule T1 T2 mu theta0 a hT1 hT2 hmu ha]
  simp [relativeMultiplierMismatch, sq_abs]

/-- Monotonicity in the actual relative mismatch: among two transfers in the
same scalar quadratic, the one with larger `|1-c2/c1|` has no smaller excess.
This is the directionally correct replacement for a false global
absolute-log-ratio claim. -/
theorem tunedTransferPenalty_mono_of_mismatch_le
    (rule : OuterRule) (T1 T2 S1 S2 : Nat) (mu theta0 a : Real)
    (hT1 : 0 < T1) (hT2 : 0 < T2) (hS1 : 0 < S1) (hS2 : 0 < S2)
    (hmu : 0 <= mu) (ha : 0 < a)
    (hmismatch : relativeMultiplierMismatch rule T1 T2 mu <=
      relativeMultiplierMismatch rule S1 S2 mu) :
    tunedTransferPenalty rule T1 T2 mu theta0 a <=
      tunedTransferPenalty rule S1 S2 mu theta0 a := by
  rw [tunedTransferPenalty_eq_mismatch_sq rule T1 T2 mu theta0 a hT1 hT2 hmu ha,
    tunedTransferPenalty_eq_mismatch_sq rule S1 S2 mu theta0 a hS1 hS2 hmu ha]
  have hleft : 0 <= relativeMultiplierMismatch rule T1 T2 mu := abs_nonneg _
  have hright : 0 <= relativeMultiplierMismatch rule S1 S2 mu := abs_nonneg _
  have hsquares : relativeMultiplierMismatch rule T1 T2 mu ^ 2 <=
      relativeMultiplierMismatch rule S1 S2 mu ^ 2 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hmismatch) (add_nonneg hright hleft)]
  exact mul_le_mul_of_nonneg_left hsquares (by positivity)

/-- The accumulated coefficient grows strictly with every added update when
momentum is nonnegative. -/
theorem effectiveCoeff_strictMono
    (rule : OuterRule) (mu : Real) (hmu : 0 <= mu) :
    StrictMono (fun T : Nat => effectiveCoeff rule T mu) := by
  apply strictMono_nat_of_lt_succ
  intro T
  unfold effectiveCoeff
  rw [Finset.sum_range_succ]
  have hterm : 0 < terminalMultiplier rule (T + 1) mu :=
    lt_of_lt_of_le (by norm_num) (one_le_terminalMultiplier_succ rule T mu hmu)
  linarith

/-- A genuine finite horizon mismatch has strictly positive transfer loss for
a nonzero scalar state. -/
theorem tunedTransferPenalty_pos_of_horizon_ne
    (rule : OuterRule) (T1 T2 : Nat) (mu theta0 a : Real)
    (hT1 : 0 < T1) (hT2 : 0 < T2) (hmu : 0 <= mu)
    (ha : 0 < a) (htheta : theta0 ≠ 0) (hTne : T1 ≠ T2) :
    0 < tunedTransferPenalty rule T1 T2 mu theta0 a := by
  have hcoeff : effectiveCoeff rule T1 mu ≠ effectiveCoeff rule T2 mu :=
    (effectiveCoeff_strictMono rule mu hmu).injective.ne hTne
  have hnonneg := tunedTransferPenalty_nonneg rule T1 T2 mu theta0 a
    hT1 hT2 hmu ha
  have hnonzero : tunedTransferPenalty rule T1 T2 mu theta0 a ≠ 0 := by
    intro hzero
    exact hcoeff ((tunedTransferPenalty_eq_zero_iff rule T1 T2 mu theta0 a
      hT1 hT2 hmu ha htheta).mp hzero)
  exact lt_of_le_of_ne hnonneg (Ne.symm hnonzero)

/-! ## Steady-state prescription -/

/-- The familiar steady-state rate prescription, applied to the no-momentum
frozen optimum at the same horizon. -/
def steadyStatePrescribedEta
    (rule : OuterRule) (T : Nat) (mu a : Real) : Real :=
  (1 - mu) * alignedOptimalEta rule T 0 a

/-- Loss excess of the steady-state prescription relative to the exact
finite-horizon aligned rate. -/
def steadyStatePrescriptionPenalty
    (rule : OuterRule) (T : Nat) (mu theta0 a : Real) : Real :=
  frozenQuadraticJ rule T mu (steadyStatePrescribedEta rule T mu a) theta0 a -
    frozenQuadraticJ rule T mu (alignedOptimalEta rule T mu a) theta0 a

/-- Exact finite-horizon error of the steady-state prescription. -/
theorem steadyStatePrescriptionPenalty_exact
    (rule : OuterRule) (T : Nat) (mu theta0 a : Real)
    (hT : 0 < T) (hmu : 0 <= mu) (ha : 0 < a) :
    steadyStatePrescriptionPenalty rule T mu theta0 a =
      (a / 2) * theta0 ^ 2 *
        (1 - effectiveCoeff rule T mu / (T : Real) * (1 - mu)) ^ 2 := by
  have hTreal : (T : Real) ≠ 0 := by exact_mod_cast hT.ne'
  have htarget :
      frozenQuadraticJ rule T mu (alignedOptimalEta rule T mu a) theta0 a = 0 := by
    simp [frozenQuadraticJ, frozenGradientQuadraticLoss,
      alignedOptimalEta_reaches_zero rule T mu theta0 a hT hmu ha]
  rw [steadyStatePrescriptionPenalty, htarget, sub_zero]
  unfold frozenQuadraticJ frozenGradientQuadraticLoss steadyStatePrescribedEta
    alignedOptimalEta
  simp only [effectiveCoeff_zero_momentum]
  field_simp [hTreal, ha.ne']

/-- The steady-state prescription becomes exact in terminal loss as the
horizon grows. -/
theorem steadyStatePrescriptionPenalty_tendsto_zero
    (rule : OuterRule) (mu theta0 a : Real)
    (hmu0 : 0 <= mu) (hmu1 : mu < 1) (ha : 0 < a) :
    Tendsto (fun T : Nat => steadyStatePrescriptionPenalty rule T mu theta0 a)
      atTop (nhds 0) := by
  have hgain := effectiveCoeff_div_mul_one_sub_tendsto_one rule mu hmu0 hmu1
  have hresidual :
      Tendsto
        (fun T : Nat => 1 - effectiveCoeff rule T mu / (T : Real) * (1 - mu))
        atTop (nhds 0) := by
    convert (tendsto_const_nhds (x := (1 : Real))).sub hgain using 1
    all_goals norm_num
  have hclosed :
      Tendsto
        (fun T : Nat =>
          (a / 2) * theta0 ^ 2 *
            (1 - effectiveCoeff rule T mu / (T : Real) * (1 - mu)) ^ 2)
        atTop (nhds 0) := by
    simpa using
      (tendsto_const_nhds.mul (hresidual.pow 2) :
        Tendsto
          (fun T : Nat =>
            ((a / 2) * theta0 ^ 2) *
              (1 - effectiveCoeff rule T mu / (T : Real) * (1 - mu)) ^ 2)
          atTop (nhds (((a / 2) * theta0 ^ 2) * 0 ^ 2)))
  apply hclosed.congr'
  filter_upwards [eventually_gt_atTop (0 : Nat)] with T hT
  exact (steadyStatePrescriptionPenalty_exact rule T mu theta0 a hT hmu0 ha).symm

/-! ## Transfer between jointly growing horizons -/

/-- For a fixed finite offset, the ratio of accumulated coefficients at
`T+d` and `T` tends to one. -/
theorem transferMultiplierRatio_fixed_offset_tendsto_one
    (rule : OuterRule) (d : Nat) (mu : Real)
    (hmu0 : 0 <= mu) (hmu1 : mu < 1) :
    Tendsto (fun T : Nat => transferMultiplierRatio rule T (T + d) mu)
      atTop (nhds 1) := by
  have havg := effectiveCoeff_div_tendsto_steady rule mu hmu0 hmu1
  have hshift :
      Tendsto
        (fun T : Nat =>
          effectiveCoeff rule (T + d) mu / ((T + d : Nat) : Real))
        atTop (nhds (steadyMultiplier mu)) := by
    change Tendsto
      ((fun T : Nat => effectiveCoeff rule T mu / (T : Real)) ∘
        (fun T : Nat => T + d)) atTop (nhds (steadyMultiplier mu))
    exact havg.comp (tendsto_add_atTop_nat d)
  have hnat : Tendsto (fun T : Nat => ((T : Real) + d) / (T : Real))
      atTop (nhds 1) := by
    have hdiv : Tendsto (fun T : Nat => (d : Real) / (T : Real))
        atTop (nhds 0) :=
      tendsto_const_nhds.div_atTop tendsto_natCast_atTop_atTop
    have hsimple : Tendsto (fun T : Nat => 1 + (d : Real) / (T : Real))
        atTop (nhds 1) := by
      convert (tendsto_const_nhds (x := (1 : Real))).add hdiv using 1
      all_goals norm_num
    apply hsimple.congr'
    filter_upwards [eventually_gt_atTop (0 : Nat)] with T hT
    have hTreal : (T : Real) ≠ 0 := by exact_mod_cast hT.ne'
    field_simp [hTreal]
  have hsteady : steadyMultiplier mu ≠ 0 := by
    unfold steadyMultiplier
    positivity
  have hratio :
      Tendsto
        (fun T : Nat =>
          (effectiveCoeff rule (T + d) mu / ((T + d : Nat) : Real) *
              (((T : Real) + d) / (T : Real))) /
            (effectiveCoeff rule T mu / (T : Real)))
        atTop (nhds 1) := by
    have hnum := hshift.mul hnat
    convert hnum.div havg hsteady using 1
    · rfl
    · field_simp [hsteady]
  apply hratio.congr'
  filter_upwards [eventually_gt_atTop (0 : Nat)] with T hT
  have hTreal : (T : Real) ≠ 0 := by exact_mod_cast hT.ne'
  have hTdreal : ((T + d : Nat) : Real) ≠ 0 := by
    exact_mod_cast (Nat.add_pos_left hT d).ne'
  unfold transferMultiplierRatio
  norm_num [Nat.cast_add]
  field_simp [hTreal, hTdreal]

/-- Consequently, transferring a tuned rate from `T` to `T+d` has vanishing
loss for every fixed offset `d`. -/
theorem tunedTransferPenalty_fixed_offset_tendsto_zero
    (rule : OuterRule) (d : Nat) (mu theta0 a : Real)
    (hmu0 : 0 <= mu) (hmu1 : mu < 1) (ha : 0 < a) :
    Tendsto (fun T : Nat => tunedTransferPenalty rule T (T + d) mu theta0 a)
      atTop (nhds 0) := by
  have hratio := transferMultiplierRatio_fixed_offset_tendsto_one
    rule d mu hmu0 hmu1
  have hresidual :
      Tendsto
        (fun T : Nat => 1 - transferMultiplierRatio rule T (T + d) mu)
        atTop (nhds 0) := by
    convert (tendsto_const_nhds (x := (1 : Real))).sub hratio using 1
    all_goals norm_num
  have hclosed :
      Tendsto
        (fun T : Nat =>
          (a / 2) * theta0 ^ 2 *
            (1 - transferMultiplierRatio rule T (T + d) mu) ^ 2)
        atTop (nhds 0) := by
    simpa using
      (tendsto_const_nhds.mul (hresidual.pow 2) :
        Tendsto
          (fun T : Nat =>
            ((a / 2) * theta0 ^ 2) *
              (1 - transferMultiplierRatio rule T (T + d) mu) ^ 2)
          atTop (nhds (((a / 2) * theta0 ^ 2) * 0 ^ 2)))
  apply hclosed.congr'
  filter_upwards [eventually_gt_atTop (0 : Nat)] with T hT
  have hTd : 0 < T + d := Nat.add_pos_left hT d
  exact (tunedTransferPenalty_exact rule T (T + d) mu theta0 a
    hT hTd hmu0 ha).symm

end

end LeanMechanism
