import LeanMechanism.QuadraticAlignment

/-!
# Dimensionless-age collapse for the registered finite-horizon law

This module formalizes the approximation behind plotting the registered
final-round deviation

`registeredFinalD T mu = 1 / (1 - mu^(T+1))`

against the dimensionless optimizer age `a = T * (1 - mu)`.  The statements
concern the deterministic scalar law already isolated in
`QuadraticAlignment`; they do not claim that stochastic training runs with
different horizons or momentum values have identical trajectories.

On `0 <= mu < 1`, the exponential surrogate has a uniform one-step Taylor
error.  Lifting that estimate to powers gives an explicit collapse error with
constant one.  For the reciprocal registered law, an age lower bound keeps its
denominators away from zero and gives a concrete constant-nine bound.  The
final section records strict order and uniqueness along each coordinate axis.
-/

namespace LeanMechanism

noncomputable section

/-- Dimensionless optimizer age for a horizon and momentum value. -/
def dimensionlessAge (T : Nat) (mu : Real) : Real :=
  (T : Real) * (1 - mu)

/-- The master curve used to approximate the registered final-round law. -/
def ageMasterCurve (a : Real) : Real :=
  1 / (1 - Real.exp (-a))

/-! ## Power-level collapse -/

/-- The one-step exponential surrogate differs from `mu` by at most
`(1 - mu)^2`.  This uniform estimate holds on the full unit interval, so no
lower bound such as `mu >= 1/2` is needed. -/
theorem exp_neg_one_sub_mu_error
    (mu : Real) (hmu : mu ∈ Set.Icc (0 : Real) 1) :
    |Real.exp (-(1 - mu)) - mu| ≤ (1 - mu) ^ 2 := by
  have hdelta0 : 0 ≤ 1 - mu := sub_nonneg.mpr hmu.2
  have hdelta1 : 1 - mu ≤ 1 := by linarith [hmu.1]
  have hx : |-(1 - mu)| ≤ (1 : Real) := by
    rw [abs_neg, abs_of_nonneg hdelta0]
    exact hdelta1
  have h := Real.abs_exp_sub_one_sub_id_le hx
  rw [show Real.exp (-(1 - mu)) - 1 - (-(1 - mu)) =
      Real.exp (-(1 - mu)) - mu by ring,
    show (-(1 - mu)) ^ 2 = (1 - mu) ^ 2 by ring] at h
  exact h

/-- A geometric power lies below its dimensionless-age exponential surrogate.
This is the power lift of `1 - x <= exp (-x)`. -/
theorem pow_le_exp_neg_dimensionlessAge
    (T : Nat) (mu : Real) (hmu : mu ∈ Set.Icc (0 : Real) 1) :
    mu ^ T ≤ Real.exp (-dimensionlessAge T mu) := by
  have hstep : mu ≤ Real.exp (-(1 - mu)) := by
    simpa only [sub_sub_cancel] using Real.one_sub_le_exp_neg (1 - mu)
  have hpow : mu ^ T ≤ Real.exp (-(1 - mu)) ^ T :=
    pow_le_pow_left₀ hmu.1 hstep T
  calc
    mu ^ T ≤ Real.exp (-(1 - mu)) ^ T := hpow
    _ = Real.exp (-dimensionlessAge T mu) := by
      rw [← Real.exp_nat_mul]
      congr 1
      simp only [dimensionlessAge]
      ring

/-- **Dimensionless-age collapse bound.**

For a nonempty horizon and `mu` in `[0,1)`, replacing `mu^T` by
`exp (-T*(1-mu))` costs at most `T*(1-mu)^2`.  Thus the explicit constant in
the requested `C*T*(1-mu)^2` form is `C = 1`; the result is stronger than the
`mu >= 1/2` regime needed by the experiments. -/
theorem pow_exp_dimensionlessAge_error
    (T : Nat) (mu : Real) (hT : 1 ≤ T) (hmu : mu ∈ Set.Ico (0 : Real) 1) :
    |mu ^ T - Real.exp (-dimensionlessAge T mu)| ≤
      (T : Real) * (1 - mu) ^ 2 := by
  have hmuIcc : mu ∈ Set.Icc (0 : Real) 1 := ⟨hmu.1, hmu.2.le⟩
  have hTreal : 0 ≤ (T : Real) := by
    have : 0 < T := by omega
    exact_mod_cast this.le
  have hmuAbs : |mu| ≤ (1 : Real) := by
    rw [abs_of_nonneg hmu.1]
    exact hmu.2.le
  have hExpPos : 0 < Real.exp (-(1 - mu)) := Real.exp_pos _
  have hExpAbs : |Real.exp (-(1 - mu))| ≤ (1 : Real) := by
    rw [abs_of_pos hExpPos, Real.exp_le_one_iff]
    linarith [hmu.2.le]
  have hmaxNonneg : 0 ≤ max |mu| |Real.exp (-(1 - mu))| :=
    le_trans (abs_nonneg mu) (le_max_left _ _)
  have hmax : max |mu| |Real.exp (-(1 - mu))| ≤ (1 : Real) :=
    max_le hmuAbs hExpAbs
  have hfactor :
      max |mu| |Real.exp (-(1 - mu))| ^ (T - 1) ≤ (1 : Real) := by
    simpa using pow_le_pow_left₀ hmaxNonneg hmax (T - 1)
  have hexpPow :
      Real.exp (-(1 - mu)) ^ T = Real.exp (-dimensionlessAge T mu) := by
    rw [← Real.exp_nat_mul]
    congr 1
    simp only [dimensionlessAge]
    ring
  rw [← hexpPow]
  calc
    |mu ^ T - Real.exp (-(1 - mu)) ^ T| ≤
        |mu - Real.exp (-(1 - mu))| * (T : Real) *
          max |mu| |Real.exp (-(1 - mu))| ^ (T - 1) :=
      abs_pow_sub_pow_le mu (Real.exp (-(1 - mu))) T
    _ ≤ (1 - mu) ^ 2 * (T : Real) *
          max |mu| |Real.exp (-(1 - mu))| ^ (T - 1) := by
      gcongr
      simpa only [abs_sub_comm] using exp_neg_one_sub_mu_error mu hmuIcc
    _ ≤ (1 - mu) ^ 2 * (T : Real) * 1 := by
      gcongr
    _ = (T : Real) * (1 - mu) ^ 2 := by ring

/-! ## Collapse of the reciprocal registered law -/

private theorem exp_neg_half_le_two_thirds :
    Real.exp (-(1 / 2 : Real)) ≤ 2 / 3 := by
  have hhalf : (3 / 2 : Real) ≤ Real.exp (1 / 2) := by
    nlinarith [Real.add_one_le_exp (1 / 2 : Real)]
  rw [Real.exp_neg]
  calc
    (Real.exp (1 / 2))⁻¹ ≤ ((3 / 2 : Real))⁻¹ :=
      inv_anti₀ (by norm_num) hhalf
    _ = 2 / 3 := by norm_num

private theorem exp_neg_le_two_thirds_of_half_le
    {a : Real} (ha : (1 / 2 : Real) ≤ a) :
    Real.exp (-a) ≤ 2 / 3 := by
  calc
    Real.exp (-a) ≤ Real.exp (-(1 / 2 : Real)) := by
      exact Real.exp_le_exp.mpr (by linarith)
    _ ≤ 2 / 3 := exp_neg_half_le_two_thirds

/-- **Registered-law age-collapse bound.**

Let `a' = (T+1)*(1-mu)`, the age matching the code-true exponent in
`registeredFinalD`.  If `a' >= 1/2`, then both reciprocal denominators are at
least `1/3`, and the power-level constant-one estimate gives the explicit
constant-nine bound below. -/
theorem registeredFinalD_ageMasterCurve_error
    (T : Nat) (mu : Real) (hT : 1 ≤ T) (hmu : mu ∈ Set.Ico (0 : Real) 1)
    (hage : (1 / 2 : Real) ≤ dimensionlessAge (T + 1) mu) :
    |registeredFinalD T mu -
        ageMasterCurve (dimensionlessAge (T + 1) mu)| ≤
      9 * (T + 1 : Real) * (1 - mu) ^ 2 := by
  have hmuIcc : mu ∈ Set.Icc (0 : Real) 1 := ⟨hmu.1, hmu.2.le⟩
  have hTsucc : 1 ≤ T + 1 := by omega
  have hy :
      Real.exp (-dimensionlessAge (T + 1) mu) ≤ (2 / 3 : Real) :=
    exp_neg_le_two_thirds_of_half_le hage
  have hxle :
      mu ^ (T + 1) ≤ Real.exp (-dimensionlessAge (T + 1) mu) :=
    pow_le_exp_neg_dimensionlessAge (T + 1) mu hmuIcc
  have hx : mu ^ (T + 1) ≤ (2 / 3 : Real) := hxle.trans hy
  have hdenx : (1 / 3 : Real) ≤ 1 - mu ^ (T + 1) := by linarith
  have hdeny : (1 / 3 : Real) ≤
      1 - Real.exp (-dimensionlessAge (T + 1) mu) := by linarith
  have hdenxPos : 0 < 1 - mu ^ (T + 1) := lt_of_lt_of_le (by norm_num) hdenx
  have hdenyPos : 0 < 1 - Real.exp (-dimensionlessAge (T + 1) mu) :=
    lt_of_lt_of_le (by norm_num) hdeny
  have hdenProd : (1 / 9 : Real) ≤
      (1 - mu ^ (T + 1)) *
        (1 - Real.exp (-dimensionlessAge (T + 1) mu)) := by
    have hmul := mul_le_mul hdenx hdeny (by norm_num : (0 : Real) ≤ 1 / 3)
      hdenxPos.le
    norm_num at hmul ⊢
    exact hmul
  have hfrac :
      |registeredFinalD T mu -
          ageMasterCurve (dimensionlessAge (T + 1) mu)| =
        |mu ^ (T + 1) - Real.exp (-dimensionlessAge (T + 1) mu)| /
          ((1 - mu ^ (T + 1)) *
            (1 - Real.exp (-dimensionlessAge (T + 1) mu))) := by
    have halgebra :
        1 / (1 - mu ^ (T + 1)) -
            1 / (1 - Real.exp (-dimensionlessAge (T + 1) mu)) =
          (mu ^ (T + 1) - Real.exp (-dimensionlessAge (T + 1) mu)) /
            ((1 - mu ^ (T + 1)) *
              (1 - Real.exp (-dimensionlessAge (T + 1) mu))) := by
      field_simp [hdenxPos.ne', hdenyPos.ne']
      all_goals ring
    rw [registeredFinalD, ageMasterCurve, halgebra, abs_div,
      abs_of_pos (mul_pos hdenxPos hdenyPos)]
  have hpower :=
    pow_exp_dimensionlessAge_error (T + 1) mu hTsucc hmu
  rw [hfrac]
  calc
    |mu ^ (T + 1) - Real.exp (-dimensionlessAge (T + 1) mu)| /
        ((1 - mu ^ (T + 1)) *
          (1 - Real.exp (-dimensionlessAge (T + 1) mu))) ≤
        |mu ^ (T + 1) - Real.exp (-dimensionlessAge (T + 1) mu)| /
          (1 / 9 : Real) :=
      div_le_div_of_nonneg_left (abs_nonneg _) (by norm_num) hdenProd
    _ = 9 * |mu ^ (T + 1) -
          Real.exp (-dimensionlessAge (T + 1) mu)| := by ring
    _ ≤ 9 * ((T + 1 : Nat) : Real) * (1 - mu) ^ 2 := by
      nlinarith [hpower]
    _ = 9 * (T + 1 : Real) * (1 - mu) ^ 2 := by
      norm_num

/-! ## Strict age order and axis-wise uniqueness -/

/-- The master curve itself is strictly decreasing at every positive age. -/
theorem ageMasterCurve_strictAntiOn :
    StrictAntiOn ageMasterCurve (Set.Ioi (0 : Real)) := by
  intro a ha b hb hab
  have hexp : Real.exp (-b) < Real.exp (-a) :=
    Real.exp_lt_exp.mpr (neg_lt_neg hab)
  have hden : 1 - Real.exp (-a) < 1 - Real.exp (-b) := by linarith
  have hdenPos : 0 < 1 - Real.exp (-a) := by
    rw [sub_pos, Real.exp_lt_one_iff]
    exact neg_lt_zero.mpr (Set.mem_Ioi.mp ha)
  unfold ageMasterCurve
  exact one_div_lt_one_div_of_lt hdenPos hden

/-- At fixed `mu < 1`, the registered age is strictly increasing with the
horizon. -/
theorem dimensionlessAge_strictMono_horizon
    (mu : Real) (hmu : mu < 1) :
    StrictMono (fun T : Nat => dimensionlessAge (T + 1) mu) := by
  intro T₁ T₂ hT
  have hcast : ((T₁ + 1 : Nat) : Real) < ((T₂ + 1 : Nat) : Real) := by
    exact_mod_cast Nat.add_lt_add_right hT 1
  unfold dimensionlessAge
  exact mul_lt_mul_of_pos_right hcast (sub_pos.mpr hmu)

/-- At fixed `0 < mu < 1`, the registered deviation is strictly decreasing
with the horizon.  This uses the existing strict antitonicity of powers below
one. -/
theorem registeredFinalD_strictAnti_horizon
    (mu : Real) (hmu : mu ∈ Set.Ioo (0 : Real) 1) :
    StrictAnti (fun T : Nat => registeredFinalD T mu) := by
  intro T₁ T₂ hT
  have hpow : mu ^ (T₂ + 1) < mu ^ (T₁ + 1) :=
    pow_right_strictAnti₀ hmu.1 hmu.2 (Nat.add_lt_add_right hT 1)
  have hdenOrder : 1 - mu ^ (T₁ + 1) < 1 - mu ^ (T₂ + 1) := by
    linarith
  have hdenPos : 0 < 1 - mu ^ (T₁ + 1) := by
    exact sub_pos.mpr (pow_lt_one₀ hmu.1.le hmu.2 (by omega))
  unfold registeredFinalD
  exact one_div_lt_one_div_of_lt hdenPos hdenOrder

/-- Along the horizon axis, increasing registered age strictly decreases `D`.
Both inequalities are exposed so the order reversal is explicit. -/
theorem registeredFinalD_strict_decreasing_age_horizon
    (mu : Real) (hmu : mu ∈ Set.Ioo (0 : Real) 1)
    {T₁ T₂ : Nat} (hT : T₁ < T₂) :
    dimensionlessAge (T₁ + 1) mu < dimensionlessAge (T₂ + 1) mu ∧
      registeredFinalD T₂ mu < registeredFinalD T₁ mu :=
  ⟨dimensionlessAge_strictMono_horizon mu hmu.2 hT,
    registeredFinalD_strictAnti_horizon mu hmu hT⟩

/-- For fixed `0 < mu < 1`, the registered deviation determines the horizon
uniquely. -/
theorem registeredFinalD_horizon_injective
    (mu : Real) (hmu : mu ∈ Set.Ioo (0 : Real) 1) :
    Function.Injective (fun T : Nat => registeredFinalD T mu) :=
  (registeredFinalD_strictAnti_horizon mu hmu).injective

/-- For every fixed horizon, registered age is strictly decreasing as momentum
increases. -/
theorem dimensionlessAge_strictAnti_momentum (T : Nat) :
    StrictAnti (fun mu : Real => dimensionlessAge (T + 1) mu) := by
  intro mu₁ mu₂ hmu
  have hsub : 1 - mu₂ < 1 - mu₁ := by linarith
  have hcast : 0 < ((T + 1 : Nat) : Real) := by positivity
  unfold dimensionlessAge
  exact mul_lt_mul_of_pos_left hsub hcast

/-- At fixed horizon, `registeredFinalD` is strictly increasing in momentum on
`[0,1)`, equivalently strictly decreasing as the registered age increases. -/
theorem registeredFinalD_strictMonoOn_momentum (T : Nat) :
    StrictMonoOn (fun mu : Real => registeredFinalD T mu)
      (Set.Ico (0 : Real) 1) := by
  intro mu₁ hmu₁ mu₂ hmu₂ hmu
  have hpow : mu₁ ^ (T + 1) < mu₂ ^ (T + 1) :=
    pow_lt_pow_left₀ hmu hmu₁.1 (by omega)
  have hdenOrder : 1 - mu₂ ^ (T + 1) < 1 - mu₁ ^ (T + 1) := by
    linarith
  have hdenPos : 0 < 1 - mu₂ ^ (T + 1) := by
    exact sub_pos.mpr (pow_lt_one₀ hmu₂.1 hmu₂.2 (by omega))
  unfold registeredFinalD
  exact one_div_lt_one_div_of_lt hdenPos hdenOrder

/-- Along the momentum axis, larger momentum means smaller registered age and
larger `D`; read in the increasing-age direction, `D` is strictly decreasing. -/
theorem registeredFinalD_strict_decreasing_age_momentum
    (T : Nat) {mu₁ mu₂ : Real}
    (hmu₁ : mu₁ ∈ Set.Ico (0 : Real) 1)
    (hmu₂ : mu₂ ∈ Set.Ico (0 : Real) 1) (hmu : mu₁ < mu₂) :
    dimensionlessAge (T + 1) mu₂ < dimensionlessAge (T + 1) mu₁ ∧
      registeredFinalD T mu₁ < registeredFinalD T mu₂ :=
  ⟨dimensionlessAge_strictAnti_momentum T hmu,
    registeredFinalD_strictMonoOn_momentum T hmu₁ hmu₂ hmu⟩

/-- For fixed horizon, the registered deviation determines momentum uniquely
on `[0,1)`. -/
theorem registeredFinalD_momentum_injOn (T : Nat) :
    Set.InjOn (fun mu : Real => registeredFinalD T mu)
      (Set.Ico (0 : Real) 1) :=
  (registeredFinalD_strictMonoOn_momentum T).injOn

end

end LeanMechanism
