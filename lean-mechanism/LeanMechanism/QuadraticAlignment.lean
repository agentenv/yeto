import LeanMechanism.FiniteHorizonOuter

/-!
# Quadratic optimum alignment

This module separates an exact dynamical model from the approximation under
which a finite-horizon learning-rate alignment law is true.

## Exact model

The loss is the scalar quadratic `L(theta) = (a / 2) * theta^2`, so the current
gradient is `a * theta`.  The buffer starts at zero.  A constant learning rate
`eta` and momentum `mu` are used for `T` outer updates.  Both `OuterRule`
variants use

`b_(t+1) = mu * b_t + a * theta_t`.

Heavy-ball applies `b_(t+1)`, while the code-true textbook-Nesterov rule applies
`a * theta_t + mu * b_(t+1)`.  Thus `(theta_t, b_t)` is exactly a two-state
linear time-invariant system.  `quadraticJ` is the terminal loss of this exact
recursion.  The definitions use real arithmetic; the production syncer's f32
rounding, stochastic gradients, changing local objectives, and asynchronous
effects are outside the model.

## The alignment theorem actually proved

For arbitrary `T`, `frozenQuadraticJ` freezes every gradient at the initial
value `a * theta_0`.  This is the constant-gradient (first-order path) model,
not the exact evolving-gradient recursion.  In this model the endpoint depends
on `eta` only through `eta * effectiveCoeff rule T mu`, and the explicit
nonnegative global optimum obeys the exact alignment identity.  Moreover every
global optimum obeys it when `theta_0 != 0`.  The exact recursion agrees with
the frozen model, and hence satisfies the same optimum theorem, for `T = 1`.
No all-`T` global-optimum claim is made for `quadraticJ`: after the gradient is
allowed to evolve, `theta_T` is a degree-`T` polynomial in `eta`, so the frozen
alignment identity is not generally exact.

## Finite-horizon ratios and correction

The accumulated-path coefficient used by terminal frozen loss is distinct from
the v3 registration's final-round heuristic
`D(T,mu) = 1 / (1 - mu^(T+1))`.  Both are formalized.  The accumulated optimum
ratio is the inverse accumulated-gain ratio and tends to `1 - mu`; separately,
the inverse registered `D` is exactly the normalized final-round multiplier.

Finally, `biasCorrectionScale age mu = 1 / (1 - mu^(age+1))` models syncer
commit `e7930ed` with one-indexed buffer age.  For code-true Nesterov it makes
each corrected multiplier exactly `1 / (1 - mu)`, and the `T`-step coefficient
exactly `T / (1 - mu)`.  Consequently the corrected momentum/no-momentum
optimal-LR ratio is the horizon-independent `1 - mu`.  The absolute frozen
optimum is `(1 - mu) / (a*T)`, so it still scales as `1/T`; "horizon-invariant"
refers to the momentum ratio (equivalently the per-update multiplier), not to
an absolute learning rate shared across different numbers of updates.
-/

namespace LeanMechanism

noncomputable section

open Filter Finset Matrix
open scoped BigOperators Matrix Topology

/-! ## Exact quadratic recursion as a 2 x 2 system -/

/-- State vector with coordinate `0 = theta` and coordinate `1 = buffer`. -/
abbrev QuadraticState := Fin 2 -> Real

/-- Initial state `(theta_0, 0)`. -/
def quadraticInitialState (theta0 : Real) : QuadraticState :=
  ![theta0, 0]

/-- Exact transition matrix for one scalar-quadratic outer update.

The heavy-ball matrix is
`[[1-eta*a, -eta*mu], [a, mu]]`.  Code-true Nesterov applies
`a*theta + mu*(mu*b + a*theta)`, giving
`[[1-eta*a*(1+mu), -eta*mu^2], [a, mu]]`.
-/
def quadraticStepMatrix
    (rule : OuterRule) (mu eta a : Real) : Matrix (Fin 2) (Fin 2) Real :=
  match rule with
  | .heavyBall =>
      !![1 - eta * a, -(eta * mu);
         a, mu]
  | .textbookNesterov =>
      !![1 - eta * a * (1 + mu), -(eta * mu ^ 2);
         a, mu]

/-- Exact state recursion, initialized with a zero momentum buffer. -/
def quadraticState
    (rule : OuterRule) (mu eta theta0 a : Real) : Nat -> QuadraticState
  | 0 => quadraticInitialState theta0
  | t + 1 => quadraticStepMatrix rule mu eta a *ᵥ
      quadraticState rule mu eta theta0 a t

@[simp]
theorem quadraticState_zero
    (rule : OuterRule) (mu eta theta0 a : Real) :
    quadraticState rule mu eta theta0 a 0 = quadraticInitialState theta0 := rfl

@[simp]
theorem quadraticState_succ
    (rule : OuterRule) (mu eta theta0 a : Real) (t : Nat) :
    quadraticState rule mu eta theta0 a (t + 1) =
      quadraticStepMatrix rule mu eta a *ᵥ quadraticState rule mu eta theta0 a t := rfl

/-- Exact matrix-power unrolling of the LTI recursion. -/
theorem quadraticState_eq_matrix_pow
    (rule : OuterRule) (mu eta theta0 a : Real) (T : Nat) :
    quadraticState rule mu eta theta0 a T =
      (quadraticStepMatrix rule mu eta a) ^ T *ᵥ quadraticInitialState theta0 := by
  induction T with
  | zero => simp
  | succ T ih =>
      simp only [quadraticState_succ, ih, Matrix.mulVec_mulVec, pow_succ']

/-- The second coordinate is exactly the common buffer update. -/
theorem quadraticState_buffer_succ
    (rule : OuterRule) (mu eta theta0 a : Real) (t : Nat) :
    quadraticState rule mu eta theta0 a (t + 1) 1 =
      a * quadraticState rule mu eta theta0 a t 0 +
        mu * quadraticState rule mu eta theta0 a t 1 := by
  cases rule <;>
    simp [quadraticState_succ, quadraticStepMatrix, Matrix.mulVec, dotProduct,
      Fin.sum_univ_two]

/-- The first coordinate spells out the two code-level direction conventions. -/
theorem quadraticState_theta_succ
    (rule : OuterRule) (mu eta theta0 a : Real) (t : Nat) :
    quadraticState rule mu eta theta0 a (t + 1) 0 =
      match rule with
      | .heavyBall =>
          quadraticState rule mu eta theta0 a t 0 - eta *
            (a * quadraticState rule mu eta theta0 a t 0 +
              mu * quadraticState rule mu eta theta0 a t 1)
      | .textbookNesterov =>
          quadraticState rule mu eta theta0 a t 0 - eta *
            (a * quadraticState rule mu eta theta0 a t 0 + mu *
              (a * quadraticState rule mu eta theta0 a t 0 +
                mu * quadraticState rule mu eta theta0 a t 1)) := by
  cases rule <;>
    simp [quadraticState_succ, quadraticStepMatrix, Matrix.mulVec, dotProduct,
      Fin.sum_univ_two] <;>
    ring

/-- Scalar deterministic quadratic loss `L(theta) = (a/2) * theta^2`. -/
def scalarQuadraticLoss (a theta : Real) : Real :=
  (a / 2) * theta ^ 2

/-- Exact terminal objective `J(rule,T,mu,eta,theta_0,a)` from the LTI recursion. -/
def quadraticJ
    (rule : OuterRule) (T : Nat) (mu eta theta0 a : Real) : Real :=
  scalarQuadraticLoss a (quadraticState rule mu eta theta0 a T 0)

/-! ## All-horizon optimum alignment in the frozen-gradient model -/

/-- Terminal quadratic after freezing all `T` gradients at `a * theta_0`. -/
def frozenQuadraticJ
    (rule : OuterRule) (T : Nat) (mu eta theta0 a : Real) : Real :=
  frozenGradientQuadraticLoss a theta0 (a * theta0)
    (effectiveCoeff rule T mu) eta

/-- Explicit frozen-gradient optimum.  It is independent of `theta_0` because
the frozen gradient is the quadratic gradient `a * theta_0`. -/
def alignedOptimalEta
    (rule : OuterRule) (T : Nat) (mu a : Real) : Real :=
  1 / (a * effectiveCoeff rule T mu)

/-- The explicit rate sends the frozen-gradient endpoint to zero. -/
theorem alignedOptimalEta_reaches_zero
    (rule : OuterRule) (T : Nat) (mu theta0 a : Real)
    (hT : 0 < T) (hmu : 0 ≤ mu) (ha : 0 < a) :
    theta0 - alignedOptimalEta rule T mu a *
      effectiveCoeff rule T mu * (a * theta0) = 0 := by
  have hc : effectiveCoeff rule T mu ≠ 0 :=
    (effectiveCoeff_pos rule T mu hT hmu).ne'
  unfold alignedOptimalEta
  field_simp [ha.ne', hc]
  ring

/-- The explicit frozen-gradient optimum is nonnegative under the model
assumptions. -/
theorem alignedOptimalEta_nonneg
    (rule : OuterRule) (T : Nat) (mu a : Real)
    (hT : 0 < T) (hmu : 0 ≤ mu) (ha : 0 < a) :
    0 ≤ alignedOptimalEta rule T mu a := by
  have hc : 0 < effectiveCoeff rule T mu := effectiveCoeff_pos rule T mu hT hmu
  unfold alignedOptimalEta
  positivity

/-- The explicit candidate is a global minimizer over all real learning rates,
and therefore also over the required nonnegative rates. -/
theorem alignedOptimalEta_is_frozen_minimizer
    (rule : OuterRule) (T : Nat) (mu theta0 a : Real)
    (hT : 0 < T) (hmu : 0 ≤ mu) (ha : 0 < a) :
    forall eta : Real,
      frozenQuadraticJ rule T mu (alignedOptimalEta rule T mu a) theta0 a ≤
        frozenQuadraticJ rule T mu eta theta0 a := by
  intro eta
  calc
    frozenQuadraticJ rule T mu (alignedOptimalEta rule T mu a) theta0 a = 0 := by
      simp [frozenQuadraticJ, frozenGradientQuadraticLoss,
        alignedOptimalEta_reaches_zero rule T mu theta0 a hT hmu ha]
    _ ≤ frozenQuadraticJ rule T mu eta theta0 a := by
      unfold frozenQuadraticJ frozenGradientQuadraticLoss
      positivity

/-- Every aligned candidate has the same accumulated displacement `1/a`. -/
theorem alignedOptimalEta_mul_effectiveCoeff
    (rule : OuterRule) (T : Nat) (mu a : Real)
    (hT : 0 < T) (hmu : 0 ≤ mu) (ha : 0 < a) :
    alignedOptimalEta rule T mu a * effectiveCoeff rule T mu = 1 / a := by
  have hc : effectiveCoeff rule T mu ≠ 0 :=
    (effectiveCoeff_pos rule T mu hT hmu).ne'
  unfold alignedOptimalEta
  field_simp [ha.ne', hc]

/-- **Quadratic frozen-gradient optimum-alignment theorem.**

For either rule, positive curvature, a nonempty horizon, and `mu` in `[0,1)`,
the displayed rates are nonnegative global minimizers and tuned rate times
accumulated multiplier is momentum-invariant.
-/
theorem frozenQuadratic_optimum_alignment
    (rule : OuterRule) (T : Nat) (mu theta0 a : Real)
    (hT : 0 < T) (hmu : mu ∈ Set.Ico (0 : Real) 1) (ha : 0 < a) :
    0 ≤ alignedOptimalEta rule T mu a ∧
      (forall eta : Real,
        frozenQuadraticJ rule T mu (alignedOptimalEta rule T mu a) theta0 a ≤
          frozenQuadraticJ rule T mu eta theta0 a) ∧
      0 ≤ alignedOptimalEta rule T 0 a ∧
      (forall eta : Real,
        frozenQuadraticJ rule T 0 (alignedOptimalEta rule T 0 a) theta0 a ≤
          frozenQuadraticJ rule T 0 eta theta0 a) ∧
      alignedOptimalEta rule T mu a * effectiveCoeff rule T mu =
        alignedOptimalEta rule T 0 a * effectiveCoeff rule T 0 := by
  have hzero : (0 : Real) ∈ Set.Ico (0 : Real) 1 := by norm_num
  refine ⟨alignedOptimalEta_nonneg rule T mu a hT hmu.1 ha,
    alignedOptimalEta_is_frozen_minimizer rule T mu theta0 a hT hmu.1 ha,
    alignedOptimalEta_nonneg rule T 0 a hT hzero.1 ha,
    alignedOptimalEta_is_frozen_minimizer rule T 0 theta0 a hT hzero.1 ha, ?_⟩
  rw [alignedOptimalEta_mul_effectiveCoeff rule T mu a hT hmu.1 ha,
    alignedOptimalEta_mul_effectiveCoeff rule T 0 a hT hzero.1 ha]

/-- In a strictly convex frozen quadratic with nonzero initial state, global
minimization is equivalent to the accumulated alignment `eta*c = 1/a`. -/
theorem frozenQuadratic_isMinimizer_iff_alignment
    (a theta0 c eta : Real) (ha : 0 < a) (htheta : theta0 ≠ 0) (hc : 0 < c) :
    (forall eta' : Real,
        frozenGradientQuadraticLoss a theta0 (a * theta0) c eta ≤
          frozenGradientQuadraticLoss a theta0 (a * theta0) c eta') ↔
      eta * c = 1 / a := by
  have hg : a * theta0 ≠ 0 := mul_ne_zero ha.ne' htheta
  have hquot : theta0 / (a * theta0) = 1 / a := by
    field_simp [ha.ne', htheta]
  simpa [hquot] using
    (frozenGradient_isMinimizer_iff_alignment
      a theta0 (a * theta0) c eta ha hg hc.ne')

/-- Any two global minimizers, not only the displayed candidates, satisfy the
momentum-invariant alignment law in the frozen-gradient quadratic model. -/
theorem frozenQuadratic_any_minimizers_align
    (rule : OuterRule) (T : Nat)
    (mu a theta0 etaMu etaZero : Real)
    (hT : 0 < T) (hmu : mu ∈ Set.Ico (0 : Real) 1)
    (ha : 0 < a) (htheta : theta0 ≠ 0)
    (hoptMu : forall eta : Real,
      frozenQuadraticJ rule T mu etaMu theta0 a ≤
        frozenQuadraticJ rule T mu eta theta0 a)
    (hoptZero : forall eta : Real,
      frozenQuadraticJ rule T 0 etaZero theta0 a ≤
        frozenQuadraticJ rule T 0 eta theta0 a) :
    etaMu * effectiveCoeff rule T mu =
      etaZero * effectiveCoeff rule T 0 := by
  have hcMu : 0 < effectiveCoeff rule T mu :=
    effectiveCoeff_pos rule T mu hT hmu.1
  have hcZero : 0 < effectiveCoeff rule T 0 :=
    effectiveCoeff_pos rule T 0 hT (by norm_num)
  have hMu : etaMu * effectiveCoeff rule T mu = 1 / a :=
    (frozenQuadratic_isMinimizer_iff_alignment
      a theta0 (effectiveCoeff rule T mu) etaMu ha htheta hcMu).1 <| by
        simpa [frozenQuadraticJ] using hoptMu
  have hZero : etaZero * effectiveCoeff rule T 0 = 1 / a :=
    (frozenQuadratic_isMinimizer_iff_alignment
      a theta0 (effectiveCoeff rule T 0) etaZero ha htheta hcZero).1 <| by
        simpa [frozenQuadraticJ] using hoptZero
  exact hMu.trans hZero.symm

/-! ## Exact-recursion result at one step -/

/-- One exact update equals one frozen-gradient update for either rule. -/
theorem quadraticState_one_theta
    (rule : OuterRule) (mu eta theta0 a : Real) :
    quadraticState rule mu eta theta0 a 1 0 =
      theta0 - eta * effectiveCoeff rule 1 mu * (a * theta0) := by
  cases rule <;>
    simp [quadraticState, quadraticInitialState, quadraticStepMatrix,
      effectiveCoeff, terminalMultiplier, geometricPrefix, Matrix.mulVec,
      dotProduct, Fin.sum_univ_two] <;>
    ring

/-- At `T=1`, exact terminal loss and frozen terminal loss coincide. -/
theorem quadraticJ_one_eq_frozenQuadraticJ
    (rule : OuterRule) (mu eta theta0 a : Real) :
    quadraticJ rule 1 mu eta theta0 a =
      frozenQuadraticJ rule 1 mu eta theta0 a := by
  rw [quadraticJ, frozenQuadraticJ, frozenGradientQuadraticLoss,
    scalarQuadraticLoss, quadraticState_one_theta]
  ring

/-- **Exact-recursion one-step optimum alignment.**  This is the unrestricted
`quadraticJ` theorem that is valid without freezing a later gradient. -/
theorem quadraticJ_one_optimum_alignment
    (rule : OuterRule) (mu theta0 a : Real)
    (hmu : mu ∈ Set.Ico (0 : Real) 1) (ha : 0 < a) :
    (forall eta : Real,
      quadraticJ rule 1 mu (alignedOptimalEta rule 1 mu a) theta0 a ≤
        quadraticJ rule 1 mu eta theta0 a) ∧
      (forall eta : Real,
        quadraticJ rule 1 0 (alignedOptimalEta rule 1 0 a) theta0 a ≤
          quadraticJ rule 1 0 eta theta0 a) ∧
      alignedOptimalEta rule 1 mu a * effectiveCoeff rule 1 mu =
        alignedOptimalEta rule 1 0 a * effectiveCoeff rule 1 0 := by
  have hall := frozenQuadratic_optimum_alignment rule 1 mu theta0 a
    (by norm_num) hmu ha
  rcases hall with ⟨_hEtaMu, hoptMu, _hEtaZero, hoptZero, halign⟩
  refine And.intro ?_ (And.intro ?_ halign)
  · simpa only [quadraticJ_one_eq_frozenQuadraticJ] using hoptMu
  · simpa only [quadraticJ_one_eq_frozenQuadraticJ] using hoptZero

/-- The all-horizon frozen alignment law cannot be promoted to the exact
evolving-gradient recursion.  Already at `T=2`, `mu=0`, `a=theta_0=1`, the
frozen aligned candidate `eta=1/2` has positive terminal loss, while the
nonnegative exact-recursion rate `eta=1` reaches zero.  This holds for both rule
constructors because they coincide at zero momentum. -/
theorem quadraticJ_two_step_frozen_alignment_counterexample
    (rule : OuterRule) :
    quadraticJ rule 2 0 1 1 1 <
      quadraticJ rule 2 0 (alignedOptimalEta rule 2 0 1) 1 1 := by
  cases rule <;>
    norm_num [quadraticJ, scalarQuadraticLoss, quadraticState,
      quadraticInitialState, quadraticStepMatrix, alignedOptimalEta,
      effectiveCoeff, terminalMultiplier, geometricPrefix, Matrix.mulVec,
      dotProduct, Fin.sum_univ_two]

/-! ## Accumulated optimum ratios and the registered final-round D -/

/-- Frozen-gradient optimum ratio expressed without curvature. -/
def accumulatedEtaRatio
    (rule : OuterRule) (T : Nat) (mu : Real) : Real :=
  effectiveCoeff rule T 0 / effectiveCoeff rule T mu

/-- The aligned candidates realize `accumulatedEtaRatio`. -/
theorem alignedOptimalEta_ratio_eq_accumulatedEtaRatio
    (rule : OuterRule) (T : Nat) (mu a : Real)
    (hT : 0 < T) (hmu : 0 ≤ mu) (ha : 0 < a) :
    alignedOptimalEta rule T mu a / alignedOptimalEta rule T 0 a =
      accumulatedEtaRatio rule T mu := by
  have hcMu : effectiveCoeff rule T mu ≠ 0 :=
    (effectiveCoeff_pos rule T mu hT hmu).ne'
  have hcZero : effectiveCoeff rule T 0 ≠ 0 :=
    (effectiveCoeff_pos rule T 0 hT (by norm_num)).ne'
  unfold alignedOptimalEta accumulatedEtaRatio
  field_simp [ha.ne', hcMu, hcZero]

/-- The accumulated optimum ratio is the inverse accumulated-gain ratio. -/
theorem accumulatedEtaRatio_eq_gainRatio_inv
    (rule : OuterRule) (T : Nat) (mu : Real) :
    accumulatedEtaRatio rule T mu =
      (effectiveCoeff rule T mu / effectiveCoeff rule T 0) ^ (-1 : Int) := by
  simp [accumulatedEtaRatio, inv_div]

/-- Closed accumulated-gain ratio for the code-true Nesterov rule. -/
def codeTrueAccumulatedGainRatio (T : Nat) (mu : Real) : Real :=
  ((T : Real) / (1 - mu) -
      mu ^ 2 * (1 - mu ^ T) / (1 - mu) ^ 2) / (T : Real)

/-- The preceding formula is exactly `C_T(mu)/C_T(0)`. -/
theorem codeTrueAccumulatedGainRatio_eq
    (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    effectiveCoeff codeTrueRule T mu / effectiveCoeff codeTrueRule T 0 =
      codeTrueAccumulatedGainRatio T mu := by
  change nesterovCoeff T mu / effectiveCoeff codeTrueRule T 0 = _
  rw [effectiveCoeff_zero_momentum,
    nesterovCoeff_closed_form T mu hmu]
  rfl

/-- Finite-`T` accumulated optimum ratio equals the inverse of the exact
accumulated-gain formula. -/
theorem codeTrue_finite_eta_ratio_eq_gainFormula_inv
    (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    accumulatedEtaRatio codeTrueRule T mu =
      (codeTrueAccumulatedGainRatio T mu) ^ (-1 : Int) := by
  rw [accumulatedEtaRatio_eq_gainRatio_inv,
    codeTrueAccumulatedGainRatio_eq T mu hmu]

/-- Accumulated frozen-gradient optimum ratios converge to the steady-state
law `1-mu`. -/
theorem accumulatedEtaRatio_tendsto_one_sub
    (rule : OuterRule) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    Tendsto (fun T : Nat => accumulatedEtaRatio rule T mu) atTop
      (nhds (1 - mu)) := by
  have havg := effectiveCoeff_div_tendsto_steady rule mu hmu0 hmu1
  have hden : 1 - mu ≠ 0 := sub_ne_zero.mpr (Ne.symm (ne_of_lt hmu1))
  have hsteady : steadyMultiplier mu ≠ 0 := by
    simp [steadyMultiplier, hden]
  have hinv := havg.inv₀ hsteady
  simpa [accumulatedEtaRatio, effectiveCoeff_zero_momentum, inv_div,
    steadyMultiplier, hden] using hinv

/-- For the code-true rule, the displayed optimal learning-rate ratio tends to
the steady-state factor `1-mu`. -/
theorem codeTrue_alignedOptimalEta_ratio_tendsto
    (mu a : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) (ha : 0 < a) :
    Tendsto
      (fun T : Nat =>
        alignedOptimalEta codeTrueRule T mu a /
          alignedOptimalEta codeTrueRule T 0 a)
      atTop (nhds (1 - mu)) := by
  apply (accumulatedEtaRatio_tendsto_one_sub codeTrueRule mu hmu0 hmu1).congr'
  filter_upwards [eventually_gt_atTop (0 : Nat)] with T hT
  exact (alignedOptimalEta_ratio_eq_accumulatedEtaRatio
    codeTrueRule T mu a hT hmu0 ha).symm

/-- The v3 registration's final-round deviation factor.  This is deliberately
not the accumulated-path ratio used by `frozenQuadraticJ`. -/
def registeredFinalD (T : Nat) (mu : Real) : Real :=
  1 / (1 - mu ^ (T + 1))

/-- The inverse registered final-round `D` is exactly the code-true terminal
multiplier normalized by its steady-state denominator. -/
theorem registeredFinalD_inv_eq_normalized_terminalMultiplier
    (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    (registeredFinalD T mu) ^ (-1 : Int) =
      terminalMultiplier codeTrueRule T mu * (1 - mu) := by
  have hmuden : 1 - mu ≠ 0 := sub_ne_zero.mpr (Ne.symm hmu)
  rw [codeTrue_terminalMultiplier_closed_form T mu hmu]
  simp [registeredFinalD, hmuden]

/-- Literal finite-to-steady multiplier ratio form of the registered law:
the ratio is the inverse of the registered final-round `D` formula. -/
theorem codeTrue_terminalMultiplier_ratio_eq_registeredFinalD_inv
    (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    terminalMultiplier codeTrueRule T mu / steadyMultiplier mu =
      (registeredFinalD T mu) ^ (-1 : Int) := by
  rw [registeredFinalD_inv_eq_normalized_terminalMultiplier T mu hmu]
  simp [steadyMultiplier]

/-- If one matches only the final-round multiplier (the v3 registration's
heuristic rather than terminal accumulated loss), the LR ratio is
`(1-mu) * D(T,mu)`. -/
theorem codeTrue_finalRound_etaRatio_eq_registered
    (T : Nat) (mu : Real) (hmu : mu ≠ 1) :
    terminalMultiplier codeTrueRule T 0 /
        terminalMultiplier codeTrueRule T mu =
      (1 - mu) * registeredFinalD T mu := by
  rw [codeTrue_terminalMultiplier_closed_form T mu hmu]
  simp [codeTrueRule, terminalMultiplier, geometricPrefix, registeredFinalD,
    div_eq_mul_inv]

/-! ## Code-true finite-horizon bias correction -/

/-- Syncer correction at one-indexed buffer `age`:
`1 / (1 - mu^(age+1))`.  Actual updates use ages `1, ..., T`. -/
def biasCorrectionScale (age : Nat) (mu : Real) : Real :=
  1 / (1 - mu ^ (age + 1))

/-- Corrected code-true multiplier at a given buffer age. -/
def correctedTerminalMultiplier (age : Nat) (mu : Real) : Real :=
  biasCorrectionScale age mu * terminalMultiplier codeTrueRule age mu

/-- Bias correction makes every code-true constant-gradient update carry the
exact steady-state multiplier. -/
theorem correctedTerminalMultiplier_eq_steady
    (age : Nat) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    correctedTerminalMultiplier age mu = steadyMultiplier mu := by
  have hpow : mu ^ (age + 1) < 1 := pow_lt_one₀ hmu0 hmu1 (by omega)
  have hpden : 1 - mu ^ (age + 1) ≠ 0 := (sub_pos.mpr hpow).ne'
  have hmuden : 1 - mu ≠ 0 := (sub_pos.mpr hmu1).ne'
  rw [correctedTerminalMultiplier, biasCorrectionScale,
    codeTrue_terminalMultiplier_closed_form age mu (ne_of_lt hmu1)]
  unfold steadyMultiplier
  field_simp [hpden, hmuden]

/-- Accumulated corrected coefficient over actual ages `1, ..., T`. -/
def correctedEffectiveCoeff (T : Nat) (mu : Real) : Real :=
  ∑ t ∈ range T, correctedTerminalMultiplier (t + 1) mu

/-- The corrected accumulated multiplier is exactly `T/(1-mu)`. -/
theorem correctedEffectiveCoeff_closed_form
    (T : Nat) (mu : Real) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    correctedEffectiveCoeff T mu = (T : Real) / (1 - mu) := by
  unfold correctedEffectiveCoeff
  simp_rw [correctedTerminalMultiplier_eq_steady _ mu hmu0 hmu1]
  simp [steadyMultiplier, div_eq_mul_inv]

/-- Equivalently, every nonempty corrected horizon has the same per-update
multiplier `1/(1-mu)`. -/
theorem correctedEffectiveCoeff_div_horizon
    (T : Nat) (mu : Real) (hT : 0 < T) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) :
    correctedEffectiveCoeff T mu / (T : Real) = steadyMultiplier mu := by
  rw [correctedEffectiveCoeff_closed_form T mu hmu0 hmu1]
  have hTreal : (T : Real) ≠ 0 := by exact_mod_cast hT.ne'
  unfold steadyMultiplier
  field_simp [hTreal]

/-- Frozen-gradient terminal loss with the corrected accumulated coefficient. -/
def correctedFrozenQuadraticJ
    (T : Nat) (mu eta theta0 a : Real) : Real :=
  frozenGradientQuadraticLoss a theta0 (a * theta0)
    (correctedEffectiveCoeff T mu) eta

/-- Explicit optimum for the corrected frozen-gradient model. -/
def correctedAlignedOptimalEta (T : Nat) (mu a : Real) : Real :=
  1 / (a * correctedEffectiveCoeff T mu)

/-- Closed form of the corrected optimum. -/
theorem correctedAlignedOptimalEta_closed_form
    (T : Nat) (mu a : Real) (hT : 0 < T)
    (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) (ha : 0 < a) :
    correctedAlignedOptimalEta T mu a =
      (1 - mu) / (a * (T : Real)) := by
  rw [correctedAlignedOptimalEta, correctedEffectiveCoeff_closed_form T mu hmu0 hmu1]
  have hTreal : (T : Real) ≠ 0 := by exact_mod_cast hT.ne'
  have hmuden : 1 - mu ≠ 0 := (sub_pos.mpr hmu1).ne'
  field_simp [ha.ne', hTreal, hmuden]

/-- The corrected explicit rate is a nonnegative global minimizer in the same
frozen-gradient quadratic model. -/
theorem correctedAlignedOptimalEta_is_minimizer
    (T : Nat) (mu theta0 a : Real) (hT : 0 < T)
    (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) (ha : 0 < a) :
    0 ≤ correctedAlignedOptimalEta T mu a ∧
      (forall eta : Real,
        correctedFrozenQuadraticJ T mu (correctedAlignedOptimalEta T mu a) theta0 a ≤
          correctedFrozenQuadraticJ T mu eta theta0 a) := by
  have hTreal : (0 : Real) < T := by exact_mod_cast hT
  have hcoeff : 0 < correctedEffectiveCoeff T mu := by
    rw [correctedEffectiveCoeff_closed_form T mu hmu0 hmu1]
    exact div_pos hTreal (sub_pos.mpr hmu1)
  have hresidual :
      theta0 - correctedAlignedOptimalEta T mu a *
        correctedEffectiveCoeff T mu * (a * theta0) = 0 := by
    unfold correctedAlignedOptimalEta
    field_simp [ha.ne', hcoeff.ne']
    ring
  constructor
  · unfold correctedAlignedOptimalEta
    positivity
  · intro eta
    calc
      correctedFrozenQuadraticJ T mu (correctedAlignedOptimalEta T mu a) theta0 a = 0 := by
        simp [correctedFrozenQuadraticJ, frozenGradientQuadraticLoss, hresidual]
      _ ≤ correctedFrozenQuadraticJ T mu eta theta0 a := by
        unfold correctedFrozenQuadraticJ frozenGradientQuadraticLoss
        positivity

/-- **Bias-corrected horizon-invariant optimum-ratio theorem.**

For every nonempty horizon, the corrected momentum optimum divided by the
corrected no-momentum optimum is exactly `1-mu`.
-/
theorem correctedAlignedOptimalEta_ratio_horizon_invariant
    (T : Nat) (mu a : Real) (hT : 0 < T)
    (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) (ha : 0 < a) :
    correctedAlignedOptimalEta T mu a /
      correctedAlignedOptimalEta T 0 a = 1 - mu := by
  rw [correctedAlignedOptimalEta_closed_form T mu a hT hmu0 hmu1 ha,
    correctedAlignedOptimalEta_closed_form T 0 a hT (by norm_num) (by norm_num) ha]
  have hTreal : (T : Real) ≠ 0 := by exact_mod_cast hT.ne'
  field_simp [ha.ne', hTreal]
  ring

end

end LeanMechanism
