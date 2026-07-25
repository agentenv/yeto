import Mathlib

/-!
# Finite-horizon outer-momentum multipliers

This file separates two quantities that agree only for a one-step horizon:

* `terminalMultiplier rule T mu` is the multiplier on a constant pseudo-gradient
  at buffer age `T`.  Heavy-ball has exponent `T`; the production/textbook
  Nesterov rule has exponent `T + 1`.
* `effectiveCoeff rule T mu` is the coefficient of the total displacement over
  `T` updates.  It is the sum of the applicable terminal multipliers at ages
  `1, ..., T`.

The production syncer first updates `b_t = mu * b_(t-1) + g_t` and then applies
`g_t + mu * b_t`.  Accordingly, `codeTrueRule` is `textbookNesterov`, while the
heavy-ball candidate remains available through the same definitions and generic
theorems.
-/

namespace LeanMechanism

noncomputable section

open Filter Finset
open scoped BigOperators Topology

/-- The two finite-horizon conventions considered by the derivation. -/
inductive OuterRule where
  | heavyBall
  | textbookNesterov
  deriving DecidableEq, Repr

/-- Lane A's code trace selects post-buffer textbook Nesterov. -/
def codeTrueRule : OuterRule := .textbookNesterov

/-- The polynomial geometric prefix `1 + mu + ... + mu^(T-1)`.

This definition is valid at `mu = 1`; division enters only in later closed-form
theorems, which explicitly assume `mu != 1`.
-/
def geometricPrefix (T : Nat) (mu : Real) : Real :=
  ∑ k ∈ range T, mu ^ k

@[simp]
theorem geometricPrefix_zero (mu : Real) : geometricPrefix 0 mu = 0 := by
  simp [geometricPrefix]

/-- Reverse-order recursion for a finite geometric prefix. -/
theorem geometricPrefix_succ (T : Nat) (mu : Real) :
    geometricPrefix (T + 1) mu = mu * geometricPrefix T mu + 1 := by
  unfold geometricPrefix
  rw [sum_range_succ]
  linear_combination -(geom_sum_mul mu T)

/-- The exact momentum-buffer recursion, with `b_0 = 0`. -/
def momentumBuffer (mu g : Real) : Nat → Real
  | 0 => 0
  | T + 1 => mu * momentumBuffer mu g T + g

@[simp]
theorem momentumBuffer_zero (mu g : Real) : momentumBuffer mu g 0 = 0 := rfl

@[simp]
theorem momentumBuffer_succ (mu g : Real) (T : Nat) :
    momentumBuffer mu g (T + 1) = mu * momentumBuffer mu g T + g := rfl

/-- A constant input unrolls the buffer into a finite geometric prefix. -/
theorem momentumBuffer_closed_form (mu g : Real) (T : Nat) :
    momentumBuffer mu g T = geometricPrefix T mu * g := by
  induction T with
  | zero => simp
  | succ T ih =>
      rw [momentumBuffer_succ, ih, geometricPrefix_succ]
      ring

/-- Final-round multiplier at buffer age `T`.

Heavy-ball applies `b_T`, hence `sum_(k < T) mu^k`.  Textbook Nesterov applies
`g + mu*b_T`, hence `sum_(k < T+1) mu^k`.
-/
def terminalMultiplier (rule : OuterRule) (T : Nat) (mu : Real) : Real :=
  match rule with
  | .heavyBall => geometricPrefix T mu
  | .textbookNesterov => geometricPrefix (T + 1) mu

/-- The `T`-update accumulated coefficient under a constant pseudo-gradient. -/
def effectiveCoeff (rule : OuterRule) (T : Nat) (mu : Real) : Real :=
  ∑ t ∈ range T, terminalMultiplier rule (t + 1) mu

/-- Named heavy-ball candidate:
`sum_(t=1..T) sum_(k=0..t-1) mu^k`. -/
def heavyBallCoeff (T : Nat) (mu : Real) : Real :=
  effectiveCoeff .heavyBall T mu

/-- Named textbook-Nesterov candidate:
`sum_(t=1..T) sum_(k=0..t) mu^k`. -/
def nesterovCoeff (T : Nat) (mu : Real) : Real :=
  effectiveCoeff .textbookNesterov T mu

theorem heavyBallCoeff_eq_sum (T : Nat) (mu : Real) :
    heavyBallCoeff T mu = ∑ t ∈ range T, geometricPrefix (t + 1) mu := by
  rfl

theorem nesterovCoeff_eq_sum (T : Nat) (mu : Real) :
    nesterovCoeff T mu = ∑ t ∈ range T, geometricPrefix (t + 2) mu := by
  rfl

/-- Zero-indexed direction applied at outer update `t`. -/
def outerDirection (rule : OuterRule) (mu g : Real) (t : Nat) : Real :=
  match rule with
  | .heavyBall => momentumBuffer mu g (t + 1)
  | .textbookNesterov => g + mu * momentumBuffer mu g (t + 1)

/-- Both direction conventions are represented by `terminalMultiplier`. -/
theorem outerDirection_closed_form (rule : OuterRule) (mu g : Real) (t : Nat) :
    outerDirection rule mu g t = terminalMultiplier rule (t + 1) mu * g := by
  cases rule with
  | heavyBall =>
      simp only [outerDirection, terminalMultiplier]
      exact momentumBuffer_closed_form mu g (t + 1)
  | textbookNesterov =>
      simp only [outerDirection, terminalMultiplier]
      rw [momentumBuffer_closed_form, geometricPrefix_succ (t + 1) mu]
      ring

/-- Total signed displacement applied by `T` constant-gradient updates. -/
def accumulatedDisplacement
    (rule : OuterRule) (T : Nat) (mu eta g : Real) : Real :=
  eta * ∑ t ∈ range T, outerDirection rule mu g t

/-- Exact finite-horizon accumulated displacement for either candidate rule. -/
theorem accumulatedDisplacement_closed_form
    (rule : OuterRule) (T : Nat) (mu eta g : Real) :
    accumulatedDisplacement rule T mu eta g = effectiveCoeff rule T mu * eta * g := by
  unfold accumulatedDisplacement effectiveCoeff
  simp_rw [outerDirection_closed_form]
  rw [← sum_mul]
  ring

/-! ## Rational finite-geometric forms -/

/-- Conventional rational form of a geometric prefix. -/
theorem geometricPrefix_eq_div (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    geometricPrefix T mu = (1 - mu ^ T) / (1 - mu) := by
  rw [geometricPrefix, geom_sum_eq hmu]
  have hleft : mu - 1 ≠ 0 := sub_ne_zero.mpr hmu
  have hright : 1 - mu ≠ 0 := sub_ne_zero.mpr (Ne.symm hmu)
  field_simp [hleft, hright]
  ring

/-- Heavy-ball accumulated coefficient as the requested sum of rational prefixes. -/
theorem heavyBallCoeff_eq_sum_div (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    heavyBallCoeff T mu =
      ∑ t ∈ range T, (1 - mu ^ (t + 1)) / (1 - mu) := by
  rw [heavyBallCoeff_eq_sum]
  apply sum_congr rfl
  intro t _ht
  exact geometricPrefix_eq_div (t + 1) mu hmu

/-- Textbook-Nesterov accumulated coefficient as rational prefixes. -/
theorem nesterovCoeff_eq_sum_div (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    nesterovCoeff T mu =
      ∑ t ∈ range T, (1 - mu ^ (t + 2)) / (1 - mu) := by
  rw [nesterovCoeff_eq_sum]
  apply sum_congr rfl
  intro t _ht
  exact geometricPrefix_eq_div (t + 2) mu hmu

/-- Heavy-ball's fully collapsed finite-horizon coefficient. -/
theorem heavyBallCoeff_closed_form (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    heavyBallCoeff T mu =
      (T : Real) / (1 - mu) - mu * (1 - mu ^ T) / (1 - mu) ^ 2 := by
  rw [heavyBallCoeff_eq_sum_div T mu hmu, ← sum_div, sum_sub_distrib]
  have hpowers : (∑ t ∈ range T, mu ^ (t + 1)) = mu * geometricPrefix T mu := by
    rw [geometricPrefix, mul_sum]
    apply sum_congr rfl
    intro t _ht
    rw [pow_succ]
    ring
  rw [hpowers, geometricPrefix_eq_div T mu hmu]
  simp only [sum_const, card_range, nsmul_eq_mul]
  have hden : 1 - mu ≠ 0 := sub_ne_zero.mpr (Ne.symm hmu)
  field_simp [hden]

/-- Textbook-Nesterov's fully collapsed finite-horizon coefficient. -/
theorem nesterovCoeff_closed_form (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    nesterovCoeff T mu =
      (T : Real) / (1 - mu) - mu ^ 2 * (1 - mu ^ T) / (1 - mu) ^ 2 := by
  rw [nesterovCoeff_eq_sum_div T mu hmu, ← sum_div, sum_sub_distrib]
  have hpowers : (∑ t ∈ range T, mu ^ (t + 2)) = mu ^ 2 * geometricPrefix T mu := by
    rw [geometricPrefix, mul_sum]
    apply sum_congr rfl
    intro t _ht
    rw [show t + 2 = 2 + t by omega, pow_add]
  rw [hpowers, geometricPrefix_eq_div T mu hmu]
  simp only [sum_const, card_range, nsmul_eq_mul]
  have hden : 1 - mu ≠ 0 := sub_ne_zero.mpr (Ne.symm hmu)
  field_simp [hden]

/-- With no momentum, both accumulated coefficients are exactly `T`. -/
@[simp]
theorem effectiveCoeff_zero_momentum (rule : OuterRule) (T : Nat) :
    effectiveCoeff rule T 0 = T := by
  cases rule <;>
    simp [effectiveCoeff, terminalMultiplier, geometricPrefix, zero_geom_sum]

@[simp]
theorem heavyBallCoeff_zero_momentum (T : Nat) : heavyBallCoeff T 0 = T := by
  simp [heavyBallCoeff]

@[simp]
theorem nesterovCoeff_zero_momentum (T : Nat) : nesterovCoeff T 0 = T := by
  simp [nesterovCoeff]

/-- The production/code-true final-round constant published by Lane A. -/
theorem codeTrue_terminalMultiplier_closed_form
    (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    terminalMultiplier codeTrueRule T mu = (1 - mu ^ (T + 1)) / (1 - mu) := by
  exact geometricPrefix_eq_div (T + 1) mu hmu

/-! ## Steady-state limit and finite-horizon deficit -/

/-- The known steady-state momentum multiplier. -/
def steadyMultiplier (mu : Real) : Real := 1 / (1 - mu)

/-- Shortfall of the final-round multiplier from its steady-state value. -/
def finiteStepDeficit (rule : OuterRule) (T : Nat) (mu : Real) : Real :=
  steadyMultiplier mu - terminalMultiplier rule T mu

/-- Exact deficit: heavy-ball has exponent `T`, Nesterov exponent `T+1`. -/
theorem finiteStepDeficit_closed_form
    (rule : OuterRule) (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    finiteStepDeficit rule T mu =
      match rule with
      | .heavyBall => mu ^ T / (1 - mu)
      | .textbookNesterov => mu ^ (T + 1) / (1 - mu) := by
  cases rule <;>
    simp only [finiteStepDeficit, steadyMultiplier, terminalMultiplier,
      geometricPrefix_eq_div _ mu hmu] <;>
    field_simp <;>
    ring

/-- Final-round multipliers converge to the known `1/(1-mu)` multiplier. -/
theorem terminalMultiplier_tendsto_steady
    (rule : OuterRule) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Tendsto (fun T : Nat => terminalMultiplier rule T mu) atTop
      (nhds (steadyMultiplier mu)) := by
  have hp : Tendsto (fun T : Nat => mu ^ T) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hmu0 hmu1
  have hone : Tendsto (fun _T : Nat => (1 : Real)) atTop (nhds 1) :=
    tendsto_const_nhds
  have hmu : mu ≠ 1 := ne_of_lt hmu1
  cases rule with
  | heavyBall =>
      simpa [terminalMultiplier, steadyMultiplier, geometricPrefix_eq_div _ mu hmu] using
        (hone.sub hp).div_const (1 - mu)
  | textbookNesterov =>
      have hp' : Tendsto (fun T : Nat => mu ^ (T + 1)) atTop (nhds 0) :=
        hp.comp (tendsto_add_atTop_nat 1)
      simpa [terminalMultiplier, steadyMultiplier, geometricPrefix_eq_div _ mu hmu] using
        (hone.sub hp').div_const (1 - mu)

/-- Equivalently, the final-round multiplier times `1-mu` tends to one. -/
theorem terminalMultiplier_mul_one_sub_tendsto_one
    (rule : OuterRule) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Tendsto (fun T : Nat => terminalMultiplier rule T mu * (1 - mu)) atTop
      (nhds 1) := by
  have hmu : 1 - mu ≠ 0 := sub_ne_zero.mpr (Ne.symm (ne_of_lt hmu1))
  simpa [steadyMultiplier, hmu] using
    (terminalMultiplier_tendsto_steady rule mu hmu0 hmu1).mul_const (1 - mu)

/-- The finite-step shortfall decreases monotonically with buffer age. -/
theorem finiteStepDeficit_antitone
    (rule : OuterRule) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Antitone (fun T : Nat => finiteStepDeficit rule T mu) := by
  apply antitone_nat_of_succ_le
  intro T
  have hmu : mu ≠ 1 := ne_of_lt hmu1
  have hden : 0 < 1 - mu := sub_pos.mpr hmu1
  have hpow (n : Nat) : mu ^ (n + 1) ≤ mu ^ n := by
    rw [pow_succ]
    exact mul_le_of_le_one_right (pow_nonneg hmu0 _) hmu1.le
  cases rule with
  | heavyBall =>
      rw [finiteStepDeficit_closed_form .heavyBall (T + 1) mu hmu,
        finiteStepDeficit_closed_form .heavyBall T mu hmu]
      exact (div_le_div_iff_of_pos_right hden).2 (hpow T)
  | textbookNesterov =>
      rw [finiteStepDeficit_closed_form .textbookNesterov (T + 1) mu hmu,
        finiteStepDeficit_closed_form .textbookNesterov T mu hmu]
      exact (div_le_div_iff_of_pos_right hden).2 (hpow (T + 1))

/-- The monotonically decreasing finite-step deficit vanishes. -/
theorem finiteStepDeficit_tendsto_zero
    (rule : OuterRule) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Tendsto (fun T : Nat => finiteStepDeficit rule T mu) atTop (nhds 0) := by
  have hp : Tendsto (fun T : Nat => mu ^ T) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one hmu0 hmu1
  have hmu : mu ≠ 1 := ne_of_lt hmu1
  cases rule with
  | heavyBall =>
      have hdiv : Tendsto (fun T : Nat => mu ^ T / (1 - mu)) atTop (nhds 0) := by
        simpa using hp.div_const (1 - mu)
      exact hdiv.congr' (Eventually.of_forall fun T =>
        (finiteStepDeficit_closed_form .heavyBall T mu hmu).symm)
  | textbookNesterov =>
      have hp' : Tendsto (fun T : Nat => mu ^ (T + 1)) atTop (nhds 0) :=
        hp.comp (tendsto_add_atTop_nat 1)
      have hdiv : Tendsto (fun T : Nat => mu ^ (T + 1) / (1 - mu)) atTop (nhds 0) := by
        simpa using hp'.div_const (1 - mu)
      exact hdiv.congr' (Eventually.of_forall fun T =>
        (finiteStepDeficit_closed_form .textbookNesterov T mu hmu).symm)

/-- Every nonempty geometric prefix is at least one when momentum is
nonnegative. -/
theorem one_le_geometricPrefix_succ (T : Nat) (mu : Real) (hmu : 0 ≤ mu) :
    1 ≤ geometricPrefix (T + 1) mu := by
  rw [geometricPrefix_succ]
  have hprefix : 0 ≤ geometricPrefix T mu := by
    unfold geometricPrefix
    exact sum_nonneg fun k _hk => pow_nonneg hmu k
  exact le_add_of_nonneg_left (mul_nonneg hmu hprefix)

/-- Each direction coefficient used by an actual update is at least one. -/
theorem one_le_terminalMultiplier_succ
    (rule : OuterRule) (T : Nat) (mu : Real) (hmu : 0 ≤ mu) :
    1 ≤ terminalMultiplier rule (T + 1) mu := by
  cases rule with
  | heavyBall => exact one_le_geometricPrefix_succ T mu hmu
  | textbookNesterov =>
      simpa only [terminalMultiplier] using one_le_geometricPrefix_succ (T + 1) mu hmu

/-- The accumulated coefficient dominates the no-momentum coefficient `T`. -/
theorem natCast_le_effectiveCoeff
    (rule : OuterRule) (T : Nat) (mu : Real) (hmu : 0 ≤ mu) :
    (T : Real) ≤ effectiveCoeff rule T mu := by
  unfold effectiveCoeff
  have hsum :
      (∑ _t ∈ range T, (1 : Real)) ≤
        ∑ t ∈ range T, terminalMultiplier rule (t + 1) mu :=
    sum_le_sum fun t _ht => one_le_terminalMultiplier_succ rule t mu hmu
  simpa only [sum_const, card_range, nsmul_eq_mul, mul_one] using hsum

/-- Literal limit for the accumulated coefficient in the question:
`effectiveCoeff(T,mu) * (1-mu)` diverges to `+infinity`.  The finite steady
multiplier is recovered only after also dividing by `T`, as proved below. -/
theorem effectiveCoeff_mul_one_sub_tendsto_atTop
    (rule : OuterRule) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Tendsto (fun T : Nat => effectiveCoeff rule T mu * (1 - mu)) atTop atTop := by
  apply tendsto_atTop_mono
    (fun T => mul_le_mul_of_nonneg_right
      (natCast_le_effectiveCoeff rule T mu hmu0) (sub_nonneg.mpr hmu1.le))
  exact (tendsto_natCast_atTop_atTop (R := Real)).atTop_mul_const (sub_pos.mpr hmu1)

/-- The average per-update multiplier tends to the same steady-state constant.

The division by `T` is essential: `effectiveCoeff` itself is an accumulated
coefficient and grows asymptotically like `T/(1-mu)`.
-/
theorem effectiveCoeff_div_tendsto_steady
    (rule : OuterRule) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Tendsto (fun T : Nat => effectiveCoeff rule T mu / (T : Real)) atTop
      (nhds (steadyMultiplier mu)) := by
  have hterminal := terminalMultiplier_tendsto_steady rule mu hmu0 hmu1
  have hshift :
      Tendsto (fun t : Nat => terminalMultiplier rule (t + 1) mu) atTop
        (nhds (steadyMultiplier mu)) :=
    hterminal.comp (tendsto_add_atTop_nat 1)
  simpa [effectiveCoeff, div_eq_mul_inv, mul_comm] using hshift.cesaro

/-- Normalized accumulated gain times `1-mu` tends to one. -/
theorem effectiveCoeff_div_mul_one_sub_tendsto_one
    (rule : OuterRule) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Tendsto
      (fun T : Nat => effectiveCoeff rule T mu / (T : Real) * (1 - mu))
      atTop (nhds 1) := by
  have hmu : 1 - mu ≠ 0 := sub_ne_zero.mpr (Ne.symm (ne_of_lt hmu1))
  simpa [steadyMultiplier, hmu] using
    (effectiveCoeff_div_tendsto_steady rule mu hmu0 hmu1).mul_const (1 - mu)

/-! ## Frozen-gradient quadratic optimum -/

/-- Quadratic loss after a frozen-gradient finite-horizon displacement.

This is intentionally the constant-gradient regime: `g` is held fixed for all
outer updates, and the quadratic is evaluated at the resulting endpoint.
-/
def frozenGradientQuadraticLoss
    (a theta g c eta : Real) : Real :=
  (1 / 2) * a * (theta - eta * c * g) ^ 2

/-- Candidate learning rate that exactly aligns the accumulated displacement. -/
def frozenGradientOptimalEta (theta g c : Real) : Real :=
  theta / (c * g)

/-- The candidate reaches the quadratic minimizer exactly. -/
theorem frozenGradientOptimalEta_reaches_zero
    (theta g c : Real) (hg : g ≠ 0) (hc : c ≠ 0) :
    theta - frozenGradientOptimalEta theta g c * c * g = 0 := by
  unfold frozenGradientOptimalEta
  field_simp [hc, hg]
  ring

/-- The explicit candidate is a global minimizer of the frozen-gradient loss. -/
theorem frozenGradientOptimalEta_is_minimizer
    (a theta g c : Real) (ha : 0 ≤ a) (hg : g ≠ 0) (hc : c ≠ 0) :
    ∀ eta : Real,
      frozenGradientQuadraticLoss a theta g c (frozenGradientOptimalEta theta g c) ≤
        frozenGradientQuadraticLoss a theta g c eta := by
  intro eta
  calc
    frozenGradientQuadraticLoss a theta g c (frozenGradientOptimalEta theta g c) = 0 := by
      simp [frozenGradientQuadraticLoss, frozenGradientOptimalEta_reaches_zero theta g c hg hc]
    _ ≤ frozenGradientQuadraticLoss a theta g c eta := by
      unfold frozenGradientQuadraticLoss
      positivity

/-- Exact optimum-alignment identity for any nonzero effective coefficient. -/
theorem frozenGradientOptimalEta_mul_coeff
    (theta g c : Real) (hg : g ≠ 0) (hc : c ≠ 0) :
    frozenGradientOptimalEta theta g c * c = theta / g := by
  unfold frozenGradientOptimalEta
  field_simp

/-- For a strictly convex quadratic, a learning rate is globally minimizing in
the frozen-gradient regime exactly when its accumulated aligned displacement
reaches the quadratic minimizer. -/
theorem frozenGradient_isMinimizer_iff_alignment
    (a theta g c eta : Real) (ha : 0 < a) (hg : g ≠ 0) (hc : c ≠ 0) :
    (∀ eta' : Real,
        frozenGradientQuadraticLoss a theta g c eta ≤
          frozenGradientQuadraticLoss a theta g c eta') ↔
      eta * c = theta / g := by
  constructor
  · intro hmin
    have hle := hmin (frozenGradientOptimalEta theta g c)
    have hnonneg : 0 ≤ frozenGradientQuadraticLoss a theta g c eta := by
      unfold frozenGradientQuadraticLoss
      positivity
    have hzero : frozenGradientQuadraticLoss a theta g c eta = 0 := by
      apply le_antisymm
      · simpa [frozenGradientQuadraticLoss,
          frozenGradientOptimalEta_reaches_zero theta g c hg hc] using hle
      · exact hnonneg
    have hsquare : (theta - eta * c * g) ^ 2 = 0 := by
      unfold frozenGradientQuadraticLoss at hzero
      rcases mul_eq_zero.mp hzero with hcoeff | hsquare
      · exact (mul_ne_zero (by norm_num) ha.ne' hcoeff).elim
      · exact hsquare
    have hresidual : theta - eta * c * g = 0 := by
      simpa using hsquare
    apply (eq_div_iff hg).2
    exact (sub_eq_zero.mp hresidual).symm
  · intro halign eta'
    have happlied : eta * c * g = theta := (eq_div_iff hg).1 halign
    have hresidual : theta - eta * c * g = 0 := sub_eq_zero.mpr happlied.symm
    calc
      frozenGradientQuadraticLoss a theta g c eta = 0 := by
        simp [frozenGradientQuadraticLoss, hresidual]
      _ ≤ frozenGradientQuadraticLoss a theta g c eta' := by
        unfold frozenGradientQuadraticLoss
        positivity

/-- The finite-horizon coefficient is positive for a nonempty horizon and
nonnegative momentum. -/
theorem effectiveCoeff_pos
    (rule : OuterRule) (T : Nat) (mu : Real) (hT : 0 < T) (hmu : 0 ≤ mu) :
    0 < effectiveCoeff rule T mu := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hT.ne'
  unfold effectiveCoeff
  rw [sum_range_succ]
  apply add_pos_of_nonneg_of_pos
  · apply sum_nonneg
    intro t _ht
    cases rule <;>
      simp only [terminalMultiplier, geometricPrefix] <;>
      exact (geom_sum_pos hmu (by omega)).le
  · cases rule <;>
      simp only [terminalMultiplier, geometricPrefix] <;>
      apply geom_sum_pos hmu <;>
      omega

/-- **Optimum-alignment law.** For either swappable rule, the explicit
loss-minimizing learning rates at momentum `mu` and at zero momentum have the
same accumulated aligned displacement.  Both rates are global minimizers by
`frozenGradientOptimalEta_is_minimizer`.
-/
theorem finiteHorizon_optimum_alignment
    (rule : OuterRule) (T : Nat) (mu a theta g : Real)
    (hT : 0 < T) (hmu : 0 ≤ mu) (ha : 0 ≤ a) (hg : g ≠ 0) :
    let cMu := effectiveCoeff rule T mu
    let cZero := effectiveCoeff rule T 0
    let etaMu := frozenGradientOptimalEta theta g cMu
    let etaZero := frozenGradientOptimalEta theta g cZero
    (∀ eta : Real,
        frozenGradientQuadraticLoss a theta g cMu etaMu ≤
          frozenGradientQuadraticLoss a theta g cMu eta) ∧
      (∀ eta : Real,
        frozenGradientQuadraticLoss a theta g cZero etaZero ≤
          frozenGradientQuadraticLoss a theta g cZero eta) ∧
      etaMu * cMu = etaZero * cZero := by
  dsimp only
  have hcMu : effectiveCoeff rule T mu ≠ 0 :=
    (effectiveCoeff_pos rule T mu hT hmu).ne'
  have hcZero : effectiveCoeff rule T 0 ≠ 0 := by
    simpa using (show (T : Real) ≠ 0 by exact_mod_cast hT.ne')
  refine ⟨frozenGradientOptimalEta_is_minimizer a theta g _ ha hg hcMu,
    frozenGradientOptimalEta_is_minimizer a theta g _ ha hg hcZero, ?_⟩
  rw [frozenGradientOptimalEta_mul_coeff theta g _ hg hcMu,
    frozenGradientOptimalEta_mul_coeff theta g _ hg hcZero]

/-- Any global minimizers, not only the explicit `frozenGradientOptimalEta`,
obey the finite-horizon optimum-alignment law. -/
theorem finiteHorizon_any_minimizers_align
    (rule : OuterRule) (T : Nat) (mu a theta g etaMu etaZero : Real)
    (hT : 0 < T) (hmu : 0 ≤ mu) (ha : 0 < a) (hg : g ≠ 0)
    (hoptMu : ∀ eta : Real,
      frozenGradientQuadraticLoss a theta g (effectiveCoeff rule T mu) etaMu ≤
        frozenGradientQuadraticLoss a theta g (effectiveCoeff rule T mu) eta)
    (hoptZero : ∀ eta : Real,
      frozenGradientQuadraticLoss a theta g (effectiveCoeff rule T 0) etaZero ≤
        frozenGradientQuadraticLoss a theta g (effectiveCoeff rule T 0) eta) :
    etaMu * effectiveCoeff rule T mu = etaZero * effectiveCoeff rule T 0 := by
  have hcMu : effectiveCoeff rule T mu ≠ 0 :=
    (effectiveCoeff_pos rule T mu hT hmu).ne'
  have hcZero : effectiveCoeff rule T 0 ≠ 0 := by
    simpa using (show (T : Real) ≠ 0 by exact_mod_cast hT.ne')
  calc
    etaMu * effectiveCoeff rule T mu = theta / g :=
      (frozenGradient_isMinimizer_iff_alignment
        a theta g (effectiveCoeff rule T mu) etaMu ha hg hcMu).1 hoptMu
    _ = etaZero * effectiveCoeff rule T 0 :=
      ((frozenGradient_isMinimizer_iff_alignment
        a theta g (effectiveCoeff rule T 0) etaZero ha hg hcZero).1 hoptZero).symm

end

end LeanMechanism
