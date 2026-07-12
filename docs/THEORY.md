# Theory: The Outer Nesterov Filter (Phase 4)

Status: 2026-07-12. Results A/B/C per docs/NORTH_STAR_PLAN.md Phase 4.
Every claim is labeled **Proposition/Theorem** (proved here, self-contained)
or **Conjecture / needs-proof** (stated, unproven); the one asymptotic
result (B.7) is labeled as a proof sketch. Sources of ground truth:
docs/OPTIMIZER_SEMANTICS.md (the update actually applied),
`syncer/src/merge.rs` (capped-Nesterov constants and code), docs/EXP2_25.md
Correction and experiment-results/EXP2/rda-rho-law/summary.md (the empirical
two-term law this theory must be consistent with). Independent derivation
checks by Codex (gpt-5.6-sol) are recorded in §4.

## 0. Setup and notation

Per fragment, the syncer applies (f32 arithmetic, buffer b_0 = 0):

    b_t = μ b_{t-1} + δ_t
    d_t = δ_t + μ b_t  =  (1+μ) δ_t + μ² b_{t-1}
    θ_t = θ_{t-1} − η d_t

δ_t is the merged pseudo-gradient (a weighted RDA merge of parameter
displacements — NOT a gradient of any single loss; this gap is Assumption
(A2)/Conjecture C.5 territory, never silently assumed away). For δ_t ≠ 0
define the realized buffer/delta geometry

    c_t = ⟨b_{t-1}, δ_t⟩ / ‖δ_t‖² ,
    r_t = ‖b_{t-1} − c_t δ_t‖ / ‖δ_t‖ ,

so b_{t-1} = c_t δ_t + b⊥ with ⟨b⊥, δ_t⟩ = 0 and ‖b⊥‖ = r_t ‖δ_t‖.

**Arithmetic model.** Every proposition in this document is a statement
about the exact-arithmetic (real-number) recursion above. The production
code implements it in f32 with f64 reductions; each rounded operation tracks
the exact one to O(ulp), but the exact equalities and inequalities below can
be violated at the rounding scale — and in adversarial regimes by more: a
parameter with |θ| ≈ 2²⁴ × |step component| absorbs that component entirely
(so an exactly-descending step can round to a loss increase), and a
subnormal-range cap can round with ~1e−4 relative error rather than 2⁻²⁴
(both exhibited concretely in the §4.2 review). We do not attempt
rounding-error theorems; "exact" and "unconditional" below always mean: for
the real-arithmetic update.

---

## 1. Result A — exact filter decomposition

These are linear-algebra identities about the exact-arithmetic form of the
update the syncer applies; they hold per commit, per fragment, with no
stochastic model, no stationarity, and no assumption on where δ_t came from.

**Proposition A.1 (per-commit decomposition, constant μ).**
For any t with δ_t ≠ 0,

    d_t = A_t δ_t + d_t⊥ ,   A_t = 1 + μ + μ² c_t ,
    ⟨d_t⊥, δ_t⟩ = 0 ,        ‖d_t⊥‖ = μ² r_t ‖δ_t‖ .

*Proof.* d_t = (1+μ)δ_t + μ² b_{t-1} = (1+μ)δ_t + μ²(c_t δ_t + b⊥)
= (1 + μ + μ² c_t) δ_t + μ² b⊥. The second term is orthogonal to δ_t by
construction and has norm μ² ‖b⊥‖ = μ² r_t ‖δ_t‖. ∎

**Proposition A.2 (varying momentum μ_t).**
If the recursion uses a per-commit momentum μ_t (as capped-Nesterov does):
b_t = μ_t b_{t-1} + δ_t, d_t = δ_t + μ_t b_t = (1+μ_t) δ_t + μ_t² b_{t-1},
then Proposition A.1 holds verbatim with

    A_t = 1 + μ_t + μ_t² c_t ,   ‖d_t⊥‖ = μ_t² r_t ‖δ_t‖ .

*Proof.* Identical substitution; nothing in the proof used μ constant. ∎

**Proposition A.3 (finite history / lag-kernel expansion).**
With b_0 = 0 and per-commit momenta μ_1, …, μ_t,

    b_{t-1} = Σ_{s=1}^{t-1} ( Π_{j=s+1}^{t-1} μ_j ) δ_s ,

and therefore, writing the lag-k cosine ρ̂_{t,k} = ⟨δ_{t-k}, δ_t⟩ /
(‖δ_{t-k}‖ ‖δ_t‖) and norm ratio n_{t,k} = ‖δ_{t-k}‖/‖δ_t‖,

    c_t = Σ_{k=1}^{t-1} ( Π_{j=t-k+1}^{t-1} μ_j ) ρ̂_{t,k} n_{t,k} ,
    A_t = 1 + μ_t + μ_t² c_t .

For constant μ the weight is μ^{k-1}: A_t = 1 + μ + μ² Σ_{k=1}^{t-1}
μ^{k-1} ρ̂_{t,k} n_{t,k}.

*Proof.* Unroll the recursion (induction on t; b_0 = 0 makes the sum exact,
not asymptotic). The c_t expression is linearity of the inner product. ∎

**Corollary A.4 (geometric kernel, finite-history correction).**
If the process is wide-sense stationary with E⟨δ_{t-k}, δ_t⟩ = σ² ρ^k and
E‖δ_t‖² = σ² (Model B.0 below), then the ratio-of-expectations aligned gain
at finite t is

    Ā_t := ( E⟨d_t, δ_t⟩ ) / ( E‖δ_t‖² )
         = 1 + μ/(1−μρ) − μ^{t+1} ρ^t / (1−μρ) ,

i.e. the infinite-history value Ā = 1 + μ/(1−μρ) minus a correction that
decays geometrically like (μρ)^t. (At μ = 0.9, ρ = 0.56 the correction is
< 1% of Ā−1 after t ≈ 7 commits.)

*Proof.* By A.3 (constant μ), E⟨d_t, δ_t⟩ = σ² (1 + μ + μ² Σ_{k=1}^{t-1}
μ^{k-1} ρ^k) and the finite geometric sum gives μ²ρ(1−(μρ)^{t-1})/(1−μρ);
then 1 + μ + μ²ρ/(1−μρ) = 1 + μ/(1−μρ). ∎

**Remark A.5 (what is exact vs. what needs concentration).**
A.1–A.3 are exact for the *realized* per-commit quantities in the
exact-arithmetic model (the syncer's logged `OuterStepStats` are their f32
realizations; see Arithmetic model in §0). Statements of the form "E[A_t] = 1 + μ/(1−μρ)" involve
the expectation of the *ratio* c_t; A.4 instead uses the ratio of
expectations, which is exact under Model B.0. E[c_t] = ρ/(1−μρ) + o(1)
requires concentration of ‖δ_t‖² (e.g. high effective dimension) — marked
**Conjecture (needs-proof)**; all downstream results use ratio-of-expectation
forms and do not depend on it.

---

## 2. Result B — correlated-input analysis

**Model B.0 (WSS pseudo-gradient with geometric kernel).**
{δ_t} ⊂ R^n is a wide-sense stationary, zero-mean process with

    E⟨δ_t, δ_s⟩ = σ² ρ^{|t−s|} ,   0 ≤ ρ < 1, σ > 0,

and the filter has reached stationarity (b initialized in the infinite past;
Corollary A.4 bounds the finite-history error). The canonical generator is
vector AR(1): δ_t = ρ δ_{t-1} + √(1−ρ²) ξ_t with ξ_t iid, E ξ = 0,
E‖ξ‖² = σ². Honest caveats: production ρ is measured on μ = 0 (open-loop)
captures; the closed-loop kernel under momentum differs; empirical lag
kernels (summary.md: lag1..4 = 0.56/0.21/0.07/0.02 at H=16) are roughly but
not exactly geometric; norms are nonstationary over training. Everything in
this section is a statement about Model B.0, offered because its stationary
moments reproduce the empirical formulas to the digit.

### B.1 Stationary filter moments

**Proposition B.1.** Under Model B.0, with μ ∈ [0,1), in stationarity:

(i) *(buffer moments)*
    E⟨b_{t-1}, δ_t⟩ = σ² ρ/(1−μρ) ,   E⟨b_t, δ_t⟩ = σ²/(1−μρ) ,
    E‖b_t‖² = σ² (1+μρ) / ((1−μ²)(1−μρ)) .

(ii) *(aligned amplification)*
    Ā(μ,ρ) := E⟨d_t, δ_t⟩ / E‖δ_t‖² = 1 + μ/(1−μρ) .

(iii) *(energy amplification)*
    A²(μ,ρ) := E‖d_t‖² / E‖δ_t‖²
             = 1 + 2μ/(1−μρ) + μ²(1+μρ)/((1−μ²)(1−μρ)) .

(iv) *(optimal scalar gain and residual energy)*
    Ā = argmin_a E‖d_t − a δ_t‖² , and the minimum is V(μ,ρ) σ² with

    V(μ,ρ) := A² − Ā² = μ⁴ (1−ρ²) / ( (1−μ²)(1−μρ)² ) .

*Proof.* (i) b_{t-1} = Σ_{k≥1} μ^{k-1} δ_{t-k} (A.3, t → ∞), so
E⟨b_{t-1}, δ_t⟩ = σ² Σ_{k≥1} μ^{k-1} ρ^k = σ²ρ/(1−μρ). For b_t include the
k = 0 term with weight 1: σ² Σ_{k≥0} μ^k ρ^k = σ²/(1−μρ). For the variance,
E‖b_t‖² = σ² Σ_{j,k≥0} μ^{j+k} ρ^{|j−k|} = σ² [ Σ_{k≥0} μ^{2k}
+ 2 Σ_{k≥0} μ^{2k} Σ_{m≥1} (μρ)^m ] = σ²/(1−μ²) · [1 + 2μρ/(1−μρ)]
= σ² (1+μρ)/((1−μ²)(1−μρ)).
(ii) E⟨d_t, δ_t⟩ = E‖δ_t‖² + μ E⟨b_t, δ_t⟩ = σ² (1 + μ/(1−μρ)).
(iii) E‖d_t‖² = E‖δ_t + μ b_t‖² = σ² + 2μ E⟨δ_t, b_t⟩ + μ² E‖b_t‖², insert
(i).
(iv) E‖d_t − a δ_t‖² is a strictly convex quadratic in a minimized at
a* = E⟨d_t, δ_t⟩/E‖δ_t‖² = Ā with value (A² − Ā²)σ². Algebra for the closed
form: with x = μ/(1−μρ), Ā² = 1 + 2x + x², so
A² − Ā² = μ²(1+μρ)/((1−μ²)(1−μρ)) − μ²/(1−μρ)²
= μ² [ (1+μρ)(1−μρ) − (1−μ²) ] / ((1−μ²)(1−μρ)²)
= μ² (μ² − μ²ρ²) / ((1−μ²)(1−μρ)²) = μ⁴(1−ρ²)/((1−μ²)(1−μρ)²). ∎

**Corollary B.2 (matches the empirical law).**
(a) Ā is exactly the corrected aligned amplification of docs/EXP2_25.md
(Correction): η_eff = η (1 + μ/(1−μρ)), not η/(1−μρ); the two agree only at
ρ = 1. Constant input (ρ → 1) gives the classical DC gain Ā → 1/(1−μ).
(b) A² is exactly the "A²_RMS" of the Correction and of
experiment-results/EXP2/rda-rho-law/summary.md, and is algebraically equal
to the equivalent form used in the controller synthesis,
(1+μ)² + 2(1+μ)μ²ρ/(1−μρ) + μ⁴(1+μρ)/((1−μ²)(1−μρ)) (expand
d_t = (1+μ)δ_t + μ²b_{t-1} instead of δ_t + μb_t). Numerically, at the
measured per-tensor RDA ρ: (μ=0.9, ρ=0.5622) → 17.64; (0.9, 0.2498) → 10.06;
(0.9, 0.3277) → 11.38; (0.5, 0.5622) → 2.99 — the summary.md table values.
(c) V is new here: sanity limits V → 0 as ρ → 1 (perfectly persistent input
accumulates nothing off-axis) and V ~ μ⁴/(1−μ²) as ρ → 0 (pure variance
accumulation); V is what a scalar LR correction can never remove (B.1(iv)).

**Remark B.3 (V ≈ transverse energy; the split is not exact).**
V decomposes as V = μ⁴ E[r_t² ‖δ_t‖²]/σ² + μ⁴ E[(c_t − c̄)²‖δ_t‖²]/σ²
(true transverse + aligned-gain fluctuation). Separating the two requires
fourth moments. For isotropic inputs in effective dimension n the
fluctuation term is O(1/n) of the total, so V ≈ stationary transverse
energy — **heuristic, needs-proof** (a Gaussian AR(1) computation would
settle it); no downstream result depends on the split.

### B.2 Quadratic dynamics: stability region

**Proposition B.4 (stability is ρ-free for additive correlated noise).**
Consider one curvature eigenmode λ > 0 of a quadratic loss L(θ) = ½ θᵀHθ
with pseudo-gradient δ_t = λ θ_{t-1} + ε_t, where ε_t is any exogenous
wide-sense-stationary process (e.g. AR(1) with parameter ρ ∈ [0,1)). Then
the mean dynamics and the second-moment dynamics of (θ_t, b_t) are stable —
and a stationary covariance exists — if and only if

    0 < η λ < 2(1+μ)/(1+2μ)     (for every eigenvalue λ of H).

In particular the stability region in (η, μ) does NOT depend on ρ.

*Proof.* Substituting δ_t into the update gives the linear system

    θ_t = (1 − η(1+μ)λ) θ_{t-1} − η μ² b_{t-1} − η(1+μ) ε_t
    b_t = λ θ_{t-1} + μ b_{t-1} + ε_t ,

i.e. x_t = M x_{t-1} + N ε_t with x = (θ, b) and
M = [[1−η(1+μ)λ, −ημ²], [λ, μ]]. ε is exogenous (its evolution does not
involve θ or b), so the joint system is block-triangular and stability of
the (θ,b) block is spectral radius spr(M) < 1; with additive WSS input the
second moments are stable under the same condition. Characteristic
polynomial p(z) = z² − Tz + D with T = 1 + μ − η(1+μ)λ and
D = μ(1 − η(1+μ)λ) + ημ²λ = μ(1 − ηλ). Jury conditions for a real 2×2:
p(1) > 0, p(−1) > 0, |D| < 1.
p(1) = 1 − T + D = η(1+μ)λ − ημλ = ηλ > 0 iff ηλ > 0.
p(−1) = 1 + T + D = 2 + 2μ − ηλ(1+2μ) > 0 iff ηλ < 2(1+μ)/(1+2μ).
|D| < 1: D = μ(1−ηλ) < μ < 1 for ηλ > 0, and D > −1 iff ηλ < (1+μ)/μ,
which is implied by ηλ < 2(1+μ)/(1+2μ) since 2(1+μ)/(1+2μ) ≤ (1+μ)/μ
⇔ 2μ ≤ 1+2μ. ∎

**Remark B.5 (what ρ does and does not control).**
Correlation does not move the divergence boundary of the linear dynamics —
the observed ρ-dependent failures (EXP2.25) are not spectral instability.
What ρ controls is (a) the realized aligned step η Ā(μ,ρ) relative to a
tuned reference (overshoot), and (b) the injected noise energy through
A²(μ,ρ) and V(μ,ρ) (stationary/transient noise floor). If pseudo-gradient
error were *multiplicative* in the gradient rather than additive, mean-square
stability would couple to ρ — **open, needs-analysis**.

### B.3 The LR-adjustment regime split

Fix (μ, ρ) and a tuned memoryless reference step η* (the LR at which μ = 0
SGD is optimal). Compare three policies, exactly at the filter level:

- **P0 (naive, fixed η):** aligned step η Ā(μ,ρ) — overshoot factor Ā vs.
  the same η at μ=0. This is the EXP2.25 grid (η = 0.175 for all μ).
- **P1 (standard 1/(1−μ) adjustment):** η = η*(1−μ), matching the constant-
  signal DC gain. Realized aligned step is η* m_std with

      m_std(μ,ρ) = (1−μ) Ā(μ,ρ) = (1−μ)(1 + μ − μρ)/(1−μρ) ≤ 1 ,

  with equality iff ρ = 1 or μ = 0. As ρ → 0, m_std → 1−μ².
- **P2 (correlation-aware):** η = η*/Ā(μ,ρ). Aligned step = η* exactly; the
  residual penalty is the energy that no scalar can remove, relative excess

      X(μ,ρ) := V/Ā² = μ⁴(1−ρ²) / ( (1−μ²) (1 + μ(1−ρ))² ) .

**Proposition B.6 (regime split, filter-level).** For every μ ∈ (0,1),
ρ ∈ [0,1): m_std < 1 strictly (P1 understeps whenever input correlation is
imperfect), while P2 matches the aligned step exactly and carries excess
energy factor 1 + X(μ,ρ). For fixed ρ, X is continuous and strictly
increasing in μ, with X(0,ρ) = 0 and X → ∞ as μ → 1. Hence for any energy
tolerance γ > 0 there is a threshold μ̄(ρ,γ) ∈ (0,1) such that for
0 < μ ≤ μ̄ correlation-aware adjustment is sufficient (aligned step exact,
excess energy ≤ γ) while the standard adjustment understeps by the strict
factor m_std < 1; and for μ > μ̄ no scalar LR policy suffices (every scalar
either misses the aligned step or carries excess energy > γ) and μ itself
must shrink.

*Proof.* m_std < 1 ⇔ (1−μ)(1+μ−μρ) < 1−μρ ⇔ μ²(1−ρ) > 0. Monotonicity of
X in μ: d/dμ log X = 4/μ + 2μ/(1−μ²) − 2(1−ρ)/(1+μ(1−ρ)) and the last term
is ≤ 2 < 4/μ for μ < 1, so the derivative is positive. Limits are immediate
from the closed form; the threshold is the intermediate value theorem. That
no scalar suffices for μ > μ̄: any scalar s applied to d_t gives aligned
step s Ā η and energy excess X unchanged relative to its own aligned step
(both scale by s²) — s trades aligned mismatch against total energy but
cannot reduce X, which already exceeds γ. ∎  (Note X is NOT monotone in ρ:
d/dρ log X = −2ρ/(1−ρ²) + 2μ/(1+μ(1−ρ)) is positive at ρ = 0 and −∞ as
ρ → 1; at μ = 0.9, X = 0.96 / 1.15 / 1.22 / 0.55 at ρ = 0 / 0.25 / 0.56 /
0.9 — a peak in between. Only μ-monotonicity is claimed or used.)

Measured instantiation (per-tensor RDA ρ from summary.md):

| (μ, ρ) | m_std (P1 aligned) | X = V/Ā² (P2 excess energy) |
|---|---:|---:|
| (0.5, 0.5622) H=16 | 0.848 | 0.038 |
| (0.5, 0.2498) H=64 | 0.786 | 0.041 |
| (0.9, 0.5622) H=16 | 0.282 | 1.215 |
| (0.9, 0.2498) H=64 | 0.216 | 1.154 |

At μ = 0.5 the correlation-aware policy is within ~4% energy of tuned SGD
while the standard adjustment understeps by 15–21% — the "P2 sufficient, P1
insufficient" regime. At μ = 0.9, X > 1 at every measured ρ: even
correlation-aware scalar adjustment carries ≥ 2.1× the energy of tuned SGD —
no LR policy suffices and momentum itself must be reduced. This is the
regime the capped controller (Result C) targets.

**Proposition B.7 (stationary excess risk, small-step limit — asymptotic).**
In the setting of B.4, with ηλ in the stability region and ε_t AR(1) with
stationary variance σ_ε², as ηλ → 0 the stationary variance is

    E[θ²] = (η σ_ε² / 2λ) · F(μ,ρ) + O(η²) ,
    F(μ,ρ) = (1+ρ) / ( (1−ρ)(1−μ) ) ,

equivalently stationary excess risk E[½λθ²] = (η σ_ε²/4) F(μ,ρ) + O(η²).

*Proof sketch (asymptotic — the limit interchange is standard but not
spelled out here; derived independently in consultation §4.1).* The closed
loop is [1 − L + ηλ L H(L)] θ_t = −η H(L) ε_t with lag-operator transfer
H(L) = (1+μ−μL)/(1−μL). As ηλ → 0 the slow closed-loop pole is
z = 1 − ηλ/(1−μ) + O((ηλ)²) and the variance integral concentrates at DC,
where the filter contributes its DC gain H(1) = 1/(1−μ) and the AR(1)
spectrum contributes Σ_k Cov(ε_t, ε_{t-k}) = σ_ε²(1+ρ)/(1−ρ); the standard
OU-limit integral yields the constant. ∎(sketch)

Note what B.7 says about the empirical law: the *stationary* penalty scales
with F(μ,ρ) — increasing in ρ, momentum entering only as the DC gain
1/(1−μ), no 1/(1−μ²) variance-accumulation pole — a structurally different
(μ,ρ)-dependence from A². So A² can only govern the loss in the transient
(few-outer-step) regime, which is exactly where the experiments live
(20–320 commits); the crossover time between the A²-regime and the
F-regime is not characterized.

**Conjecture B.8 (the loss bridge — needs-proof).**
The map from (aligned mismatch, excess energy) to eval-loss penalty used
empirically — loss = c_H + b·log(η_eff/η*)² + v·log A² with (b, v) =
(0.106, 0.023), R² = 0.90 on the 9-cell grid (summary.md) — is a calibrated
regression, not a theorem. Unproven steps: (1) that per-step energy governs
short-run loss via a smoothness penalty of the form (L_s η²/2) A² σ² summed
over T outer steps (descent-lemma heuristic; plausible for the transient
regime, and B.7 shows the stationary regime obeys a different law, so the
A² claim cannot extend to long horizons unmodified); (2) the
log-quadratic/log-linear functional form; (3) closed-loop ρ = open-loop ρ;
(4) identifiability of (b, v) as separate mechanisms (leverage p/n = 5/9,
one seed; see controller-decision notes). The decisive experiment is
preregistered: matched-η_eff pairs at fixed H with different (μ, ρ) —
aligned-only predicts no difference, two-term predicts v·Δlog A².

---

## 3. Result C — safety of capped-Nesterov

Frozen controller (`syncer/src/merge.rs`, constants compile-time): μ_max =
0.9, τ⊥ = 1.0, r_eps = 1e−12, one-sided release EMA β = 0.9, initial
μ = μ_max. Per commit, from the realized geometry (c_t, r_t):

    μ_par  = largest μ ∈ [0, μ_max] with μ + μ² [c_t]₊ ≤ μ_max
           = (√(1 + 4[c_t]₊ μ_max) − 1) / (2[c_t]₊)   (μ_max if [c_t]₊ = 0)
    μ_perp = √( τ⊥ / max(r_t, r_eps) )
    cap    = min(μ_max, μ_par, μ_perp), zeroed if 1 + cap + cap² c_t < 0
    μ_t    = min( cap, β μ_{t-1} + (1−β) cap )
    b_t = μ_t b_{t-1} + δ_t ;  d_t = δ_t + μ_t b_t ;  θ −= η d_t .

Two code-fidelity notes from the §4.2 review. (F1) The implemented μ_par
formula (√(1+4cμ_max)−1)/(2c) cancels catastrophically for small positive
c: at c ≈ 2e−16 it returns ≈ 0.555 instead of ≈ 0.9, and at c ≈ 1e−20 it
returns exactly 0, so the implemented cap is discontinuous at c → 0⁺.
This is *conservative* (never exceeds the true root, so all safety bounds
below still hold for the code) but violates the "largest admissible μ"
spec; the algebraically identical stable form is
2μ_max/(1 + √(1+4cμ_max)). (F2) A checkpoint restore keeps the buffer but
resets the EMA scalar to μ_max (`state.rs`), so the smooth-release
behavior is not continuous across restores (the per-commit caps, which
carry all the safety content, bind identically on the first post-restore
commit; only the release smoothing differs — e.g. a guard-suppressed
μ = 0 becomes μ = cap after a restore).

**Lemma C.0 (controller-state invariant).** Along any production
trajectory, μ_{t-1} ∈ [0, μ_max] at every commit. *Proof.* Both entry
points set it to μ_max (fresh construction; checkpoint restore). Every
update writes μ_t = min(cap, βμ_{t-1} + (1−β)cap) with cap ∈ [0, μ_max]
(each of μ_max, μ_par, μ_perp is ≥ 0, the guard writes 0), so μ_t ≤ cap
≤ μ_max, and μ_t ≥ 0 because both arguments of the min are ≥ 0 when
μ_{t-1} ≥ 0. Induction. ∎  The invariant is load-bearing: for a corrupted
or hand-supplied μ_{t-1} < 0 the code takes the negative EMA branch and
every bound below fails (§4.2 finding 4). C.1 assumes it.

**Proposition C.1 (per-commit bounds under the C.0 invariant).**
For every commit with δ_t ≠ 0 and μ_{t-1} ∈ [0, μ_max] (Lemma C.0),
whatever the buffer b_{t-1} and the data, the applied step satisfies, with
A_t and d_t⊥ as in Proposition A.2:

    (i)   0 ≤ A_t ≤ 1 + μ_max ,
    (ii)  ‖d_t⊥‖ ≤ τ⊥ ‖δ_t‖ ,
    (iii) ‖d_t‖² = A_t²‖δ_t‖² + ‖d_t⊥‖² ≤ ((1+μ_max)² + τ⊥²) ‖δ_t‖² ,

(numerically ‖d_t‖ ≤ √4.61 ‖δ_t‖ ≈ 2.147 ‖δ_t‖ at the frozen constants).

*Proof.* By Lemma C.0, 0 ≤ μ_t ≤ cap.

(i) Upper bound. If c_t ≥ 0: g(μ) = μ + μ²c_t is increasing on μ ≥ 0. For
c_t > 0 the positive root of g(μ) = μ_max is μ_par =
2μ_max/(1+√(1+4c_tμ_max)) < μ_max (so the min with μ_max never binds
there), and μ_t ≤ cap ≤ μ_par gives g(μ_t) ≤ g(μ_par) = μ_max; for
c_t = 0, g(μ_t) = μ_t ≤ μ_max. Hence A_t = 1 + g(μ_t) ≤ 1 + μ_max.
If c_t < 0: A_t = 1 + μ_t + μ_t² c_t < 1 + μ_t ≤ 1 + μ_max.
Lower bound. If c_t ≥ 0, A_t ≥ 1. If c_t < 0, A(μ) = 1 + μ + c_t μ² is
concave in μ with A(0) = 1 > 0 and A(cap) ≥ 0 (if the guard fired, cap = 0
and A(0) = 1; otherwise the guard's test passed). A concave function on
[0, cap] lies above its chord, so A(μ_t) ≥ min(A(0), A(cap)) ≥ 0 for
μ_t ∈ [0, cap].

(ii) ‖d_t⊥‖ = μ_t² r_t ‖δ_t‖ ≤ μ_perp² r_t ‖δ_t‖ = (τ⊥ / max(r_t, r_eps))
r_t ‖δ_t‖ ≤ τ⊥ ‖δ_t‖.

(iii) Pythagoras on A.2 plus (i) and (ii). ∎

*Caveats to C.1, stated exactly:* (a) **zero-delta commits are excluded,
and the gap is operational, not measure-zero**: if δ_t = 0 the code sets
(c_t, r_t) = (0, 0), the caps are inert, and the applied step is
d_t = μ_t² b_{t-1} — pure history with no δ scale to cap against, and
unbounded relative to any gradient. Exact-zero fragment deltas can arise
from frozen or sparse parameters, unchanged fragments, quantization, or
cancellation — not only "all learners exactly at anchor" — and a restore
makes it worse (buffer kept, μ reset to μ_max). A code guard skipping the
step at δ_t = 0 would close this. (b) f32: μ_t is computed in f64 and cast
once to f32; round-to-nearest is usually ≤ 2⁻²⁴ relative but can reach
~1e−4 relative in the subnormal range, and can round *across* the guard's
root (an exact A(μ_t) = 0 boundary case becoming A ≈ −7e−8) — see the
Arithmetic model in §0; all C.1 inequalities are exact-arithmetic
statements. (c) The bounds are per fragment (buffers are fragment-scoped);
the whole-model step satisfies (iii) summed over fragments, but A_t differs
per fragment.

**Theorem C.2 (one-step descent).**
Fix a commit with δ_t ≠ 0 satisfying the Lemma C.0 invariant, and write
θ' = θ − η d_t, g = ∇L(θ) (exact arithmetic throughout, §0). Assume:

  (A1) L is L_s-smooth on the segment [θ, θ'];
  (A2) bounded relative pseudo-gradient error:
       ‖δ_t − g‖ ≤ ε ‖g‖ with ε < 1 (hence ⟨g, δ_t⟩ ≥ (1−ε)‖g‖² and
       (1−ε)‖g‖ ≤ ‖δ_t‖ ≤ (1+ε)‖g‖), g ≠ 0.

Then

    L(θ') ≤ L(θ) − η ‖g‖² [ A_t(1−ε) − ε τ⊥ (1+ε)
                             − (η L_s / 2)(A_t² + τ⊥²)(1+ε)² ] ,

and the step is a strict descent step whenever the bracket is positive. In
particular, if additionally c_t ≥ 0 (so A_t ∈ [1, 1+μ_max] by C.1), the
bracket is positive for every realizable A_t whenever

    η < ( 2 / (L_s (1+ε)²) ) · min_{A ∈ {1, 1+μ_max}}
        ( A(1−ε) − ε τ⊥ (1+ε) ) / ( A² + τ⊥² ) .

*Proof.* Descent lemma under (A1): L(θ') ≤ L(θ) − η⟨g, d_t⟩ +
(η²L_s/2)‖d_t‖². Decompose via A.2. Aligned term: ⟨g, A_t δ_t⟩ =
A_t ⟨g, δ_t⟩ ≥ A_t (1−ε) ‖g‖², using A_t ≥ 0 (C.1(i)). Transverse term:
since ⟨δ_t, d_t⊥⟩ = 0, ⟨g, d_t⊥⟩ = ⟨g − δ_t, d_t⊥⟩ ≥ −ε‖g‖ · τ⊥‖δ_t‖ ≥
−ε τ⊥ (1+ε) ‖g‖² — note the transverse budget only couples to the descent
direction through the pseudo-gradient *error*, which is why the coefficient
is ε τ⊥ and not τ⊥. Energy: ‖d_t‖² ≤ (A_t² + τ⊥²)(1+ε)²‖g‖² by C.1(iii)
before maximizing A_t. Collect. For the last claim: the bracket, viewed as
a function of A on [1, 1+μ_max] at fixed η, is concave (quadratic with
negative leading coefficient), so its minimum is at an endpoint; requiring
positivity at both endpoints gives the stated η range. ∎

**Corollary C.3 (comparison with memoryless SGD at the design scaling).**
Suppose memoryless SGD at reference LR η* satisfies the descent-lemma
guarantee with margin κ₀ ∈ (0,1]:

    (η* L_s / 2)(1+ε)² ≤ (1 − κ₀)(1−ε)

(κ₀ = 1 − the fraction of the SGD descent threshold used; κ₀ > 0 iff SGD
at η* provably descends with slack). Run capped-Nesterov at the frozen
design scaling η = η*/(1+μ_max) (the production choice 0.147 ≈ 0.28/1.9).
If c_t ≥ 0, a sufficient condition for strict one-step descent is

    (1−ε) [ A − (1−κ₀)(A² + τ⊥²)/(1+μ_max) ] > ε τ⊥ (1+ε)
        at both endpoints A = 1 and A = 1 + μ_max .

At ε = 0 this reduces to κ₀ > τ⊥² / ((1+μ_max)² + τ⊥²) = 1/4.61 ≈ 0.217
at the frozen constants; at ε = 0.2 the requirement is κ₀ ≳ 0.34.

*Proof.* Substitute η = η*/(1+μ_max) and the margin inequality into the
bracket of C.2; concavity in A reduces to endpoints. At ε = 0 the binding
endpoint is A = 1+μ_max: κ₀(1+μ_max) > (1−κ₀) τ⊥²/(1+μ_max). ∎

*Interpretation, stated honestly:* "capped-Nesterov descends whenever
memoryless SGD does" is **not** true margin-free at equal effective step.
When A_t = 1+μ_max the aligned step already equals SGD's at η*, and the
transverse budget τ⊥²‖δ‖² adds smoothness penalty with zero aligned payoff;
the theorem therefore needs SGD to hold with ≈ 22% slack (ε = 0). What the
cap buys is that this is a *constant* (ρ- and history-independent) margin:
uncapped Nesterov at μ = 0.9, ρ = 0.56 has realized aligned gain Ā ≈ 2.8
and energy A² ≈ 17.6 — no fixed margin on SGD survives that.

**Proposition C.4 (bounded one-step harm, no alignment assumption on
history).** Under (A1)–(A2), δ_t ≠ 0, and the Lemma C.0 invariant only
(c_t arbitrary, including the anti-aligned case where descent is not
guaranteed):

    L(θ') − L(θ) ≤ η ‖g‖² [ ε τ⊥ (1+ε) ]
                   + (η² L_s / 2) ((1+μ_max)² + τ⊥²)(1+ε)² ‖g‖² .

*Proof.* In the C.2 bound, drop the aligned term (−η A_t⟨g,δ_t⟩ ≤ 0 since
A_t ≥ 0 and ⟨g,δ_t⟩ ≥ (1−ε)‖g‖² ≥ 0) and maximize the rest via C.1. ∎

So even when the buffer anti-aligns (c_t < 0, where A_t may reach 0 and no
descent claim is possible), a single capped step raises the loss by at most
O(ε η + η² L_s) ‖g‖² — the same order as a memoryless SGD step with an
ε-corrupted gradient, whereas uncapped momentum has no such bound.

**What Result C does NOT claim.** (1) No multi-step or convergence
guarantee: the theorem is per-commit; nothing controls the closed-loop
distribution of (c_t, r_t) across commits. (2) (A2) is a *modeling
assumption* about merged RDA parameter displacements, not a verified
property — δ_t is a multi-step displacement, not ∇L(θ) for any fixed L;
whether production deltas satisfy (A2) with usable ε, and against which
loss, is **Conjecture C.5 (needs-proof/measurement)** — it is the load-
bearing unproven step of the safety story. (3) The c_t < 0 case yields only
C.4's bounded harm, not descent. (4) Per-fragment scoping: C.2 as stated is
per fragment. The whole-model statement does NOT follow from a global
relative-error bound (a global ε can hide a fragment with ⟨g_p, δ_p⟩ < 0
whose large realized A_p reverses the weighted aligned term — §4.2 finding
9); it needs, for every updated fragment p: (A2) with a common ε against
g_p = fragment block of ∇L, δ_p ≠ 0, c_p ≥ 0 for the endpoint argument,
plus one L_s for the combined step. Then the inner-product and energy
terms sum fragment-wise. (5) Zero-delta commits are outside C.1–C.4 (see
C.1 caveat (a)), and a single zero-delta fragment with nonzero buffer
breaks the whole-model summation in (4). (6) All guarantees are
exact-arithmetic (§0); the f32 step can violate them at rounding scale.
(7) The release-EMA narrative is not continuous across checkpoint restores
(F2); the per-commit caps are.

---

## 4. Independent derivation checks (Codex gpt-5.6-sol, 2026-07-12)

Two `codex exec -m gpt-5.6-sol -s read-only` consultations (xhigh
reasoning). Transcripts: /tmp/codex_consult_B.log, /tmp/codex_consult_C.log
(session ids 019f577c-2dae…, 019f577c-48fc…).

### 4.1 Independent derivation of Result B (no file access)

Codex was given only the filter recursion and Model B.0 (not this
document) and asked to derive the stationary moments and quadratic-mode
stability from scratch. **Verdict: full agreement on every closed form** —
E⟨b_{t-1},δ_t⟩ = σ²ρ/(1−μρ); E‖b_t‖² = σ²(1+μρ)/((1−μ²)(1−μρ));
Ā = (1+μ−μρ)/(1−μρ); A² (its single-fraction form
(1+2μ−μρ−2μ³+2μ³ρ)/((1−μ²)(1−μρ)) expands to B.1(iii)); a* = Ā; V =
μ⁴(1−ρ²)/((1−μ²)(1−μρ)²); stability region 0 < ηλ < 2(1+μ)/(1+2μ) via the
same Jury conditions (T = (1+μ)(1−ηλ), D = μ(1−ηλ)), explicitly
ρ-independent by block-triangularity. It additionally derived the
small-step stationary variance constant F(μ,ρ) = (1+ρ)/((1−ρ)(1−μ)),
adopted here as Proposition B.7 (asymptotic) and used to sharpen
Conjecture B.8's honesty about the transient-vs-stationary regimes.

### 4.2 Adversarial review of Result C (with code access)

Codex read this document and `merge.rs`/`state.rs` and attacked C.1–C.4.
Ten findings; disposition:

1. (Critical) C.2/C.4 are false for the *f32* update — a component with
   |θ| ≈ 2²⁴|step| absorbs the step, so an exactly-descending step can
   round to a loss increase (explicit 2-D counterexample). → Adopted: the
   Arithmetic model paragraph (§0) scopes every result to exact
   arithmetic; caveat (6) in Result C.
2. (High) A.1–A.2 are likewise not exact for the f32 code. → Same fix.
3. (High) The implemented μ_par root formula cancels catastrophically for
   small c₊ (0.555 at c≈2e−16; exactly 0 at c≈1e−20; discontinuous at
   c → 0⁺). Conservative for safety, but contradicts the "largest
   admissible μ" spec. → Documented as F1 with the stable formula; code
   fix listed in §5.
4. (High) "any previous μ" was too strong: μ_prev < 0 (corrupted state)
   breaks every bound via the EMA. → Adopted: Lemma C.0 invariant, C.1
   restated to assume it.
5. (High) The f32 cast can exceed caps by ~1e−4 relative in the subnormal
   range and can cross the guard root. → Adopted into C.1 caveat (b) and
   §0 (replacing the wrong "≤ 2⁻²⁴, quantitatively irrelevant" claim).
6. (Medium) Concavity argument itself valid; the hole was only μ_t ∈
   [0, cap] membership. → Closed by Lemma C.0 + §0.
7. (Medium) Restore resets the EMA scalar but keeps the buffer; caps
   still bind pointwise but release smoothing is discontinuous (worked
   example: transverse contribution 0.0081‖δ‖ vs 0.81‖δ‖ — both ≤ τ⊥‖δ‖).
   → Documented as F2 and caveat (7).
8. (Medium) Zero-delta commits are an operational gap (frozen/sparse
   fragments, quantization, cancellation), not measure-zero; restore
   worsens it. → C.1 caveat (a) rewritten; code guard in §5.
9. (Medium) Whole-model summation needs per-fragment hypotheses; a global
   ε does not suffice (explicit 2-fragment counterexample). → "does NOT
   claim" item (4) rewritten.
10. (Low/verified) Everything else checked clean: C.2 bracket concavity
    and endpoint reduction, the ε·τ⊥ transverse coupling, the energy
    bound, C.3's κ₀ thresholds (0.2169 at ε=0; 0.3406 at ε=0.2), μ_par
    root < μ_max for c > 0 (the proof's "root exceeds μ_max" branch was
    impossible — deleted), and the code's ordering (effective μ_t
    computed before the buffer update and used consistently).

---

## 5. Open problems (consolidated)

1. **Conjecture B.8** — the transient loss bridge: derive T-step expected
   excess loss on a quadratic under Model B.0 and show it is monotone in
   (log aligned mismatch)² and log A² in the 20–320-commit regime;
   characterize the crossover time to the stationary F(μ,ρ) regime of
   Proposition B.7 (whose proof should also be completed beyond the
   asymptotic sketch).
2. **Conjecture C.5** — pseudo-gradient error model: measure ‖δ_t − ∇L̂‖ /
   ‖∇L̂‖ against a held-out minibatch gradient at the anchor; the descent
   guarantee is vacuous without a usable ε.
3. Remark A.5 — concentration of realized c_t around ρ/(1−μρ).
4. Remark B.3 — exact transverse/fluctuation split (Gaussian AR(1), finite
   dimension).
5. Remark B.5 — stability under multiplicative pseudo-gradient error.
6. Closed-loop ρ: all measured kernels are open-loop (μ = 0); the model
   with momentum in the loop changes the kernel of δ_t itself.
7. Code fixes surfaced by §4.2 (behavior-changing — need the usual
   preview/bit-identity treatment, not silent edits): (a) zero-delta
   guard (skip the step when δ_t = 0; C.1 caveat (a)); (b) replace the
   μ_par root formula with the stable 2μ_max/(1+√(1+4c₊μ_max)) (F1);
   (c) optionally clamp μ_prev to [0, μ_max] on entry (defense in depth
   for Lemma C.0).
8. Rounding-robust versions of C.1–C.4 (f32 error analysis), or a
   documented decision that exact-arithmetic scope is enough.
