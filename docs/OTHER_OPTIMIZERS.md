# Other High-Odds Outer Optimizers (Codex brainstorm, 2026-07-13)

Candidates beyond the CTTN oracle, ranked by P(clears the product gate: paired
win >0.018 on ≥2 workloads, never worse than 0.009 anywhere, <1% overhead, no
extra forward). Refs vs SGD-0.28 (docs/BAKEOFF_RESULTS.md).

## 1. Secant-gated Chebyshev-SGD — P≈30% (best PRODUCT candidate)
Memoryless Chebyshev polynomial acceleration (stabilized-explicit method for
stiff spectra) applied to the merged delta, **guarded** by a cheap low-rank
secant curvature estimate: if the recent secants (Δg vs Δθ over the last ~4-5
rounds) do not predict the newest secant, fall back to exact SGD-0.28. No extra
forward/HVP, no directional memory, sub-1% overhead, exact-SGD fallback.
- Base `cheb-sgd` is ALREADY implemented (syncer/src/merge.rs, commit fea1978);
  the NEW piece is the secant guard (raises ~17% unguarded → ~30%).
- **Decisive exp:** paired {SGD, cheb-guard, cheb-guard with lag-shuffled Δg}
  on H16/H64/H256. The shuffled arm MUST lose by ≥0.009 to the real one (proves
  identified dynamics, not generic perturbation). Kill if <0.018 at both H16/H64.

## 2. Trust-Krylov (secant-based low-rank) — P≈30% (conditional on CTTN)
Low-rank SECANT Rayleigh fit + trust region + SGD descent-angle floor. Scale-
correct where wsub was fatally quartic — secants satisfy Δg≈JΔθ so their Rayleigh
info is quadratic/dimensional, not the quartic-in-delta E_b. Falls back to SGD
when secants don't predict. ~15-25 adapter-vector ops/round, 0.3-0.9% at H16, NO
forward/HVP; storage ~8-10 LoRA-sized vectors.
- P is 35-40% if CTTN preserves useful flat-mode signal; <10% if CTTN degenerates.
- **Decisive exp:** {SGD, trust-Krylov, trust-Krylov with lag-shuffled Δg} on
  H16/H64/H256; real must beat shuffled by ≥0.009.

## 3. Muon-inner × memoryless outer — P≈20% (~12% for a true OUTER win)
Muon reshapes the pseudo-gradient BEFORE the lossy DiLoCo endpoint compression
(vs Iso-C's failed post-hoc flattening). Newton-Schulz per inner step; ~1%
overhead on LoRA (measure). No outer-direction memory, no extra forward.
- **Decisive exp:** 2×2 factorial {AdamW,Muon}×{SGD-0.28, guarded-Chebyshev} on
  H16/H64/H256/inner-lr-hi. Key stat = OUTER INTERACTION
  (Muon+Cheb − Muon+SGD) − (AdamW+Cheb − AdamW+SGD); kill "Muon unlocks a better
  outer" if <0.009. Ship Muon+SGD if it independently clears the gate (label as
  an inner win, not an outer win).

## 4. Token-time tail primal averaging — P≈14% (CHEAP to test NOW)
Shadow token-weighted Polyak-Ruppert average of committed global models over the
final 25% of tokens; final blend θ_out = 0.5·θ_T + 0.5·θ_tail. Shadow-only —
never perturbs training. <0.1% cost, no extra training forward.
- Token-weighted (NOT commit-weighted) or H16 contributes 16× more states than
  H256 → hidden horizon-dependent algorithm.
- **Decisive exp:** reconstruct from EXISTING saved SGD-0.28 anchor trajectories
  (no retraining if checkpoint coverage permits); eval H16/H64/H256/inner-lr-hi.
  Kill if <0.009 win at H16. Live Lookahead/train-through-EMA only ~5%.

## 5. SCAFFOLD-lite inner control variates + SGD — P≈8%
Endpoint-derived control variates: each worker keeps c_i, server keeps mean c,
next window applies grad_i − c_i + c (updated from already-computed gradients →
no extra forward). Outer stays SGD-0.28. Reduces client drift (crossover-safe:
small at H16, larger at H256).
- Risk: strict quorum already averages all 4 workers → likely too little
  correctable bias under IID data (worker-SNR's best short-H gain was 0.0019).
- **Decisive exp:** {SCAFFOLD-lite, SGD} on H64/H256/inner-lr-hi + one
  heterogeneous-data workload. Kill if it doesn't win H256 AND heterogeneous by
  >0.018, or H16 >0.009 worse.

## The right regime signal (for any switching scheme)
NOT ρ (inner-LR spread +0.163 at fixed applied norm), NOT delta-norm, NOT scalar
Barzilai-Borwein. Diagnostic-only shadow buffer + secant Hessian:
```
bhat_t = 0.9 bhat_{t-1} + g_t              # never applied
r_t    = P_perp(g_t) bhat_{t-1}
B_t    = PSD(sym(secant_fit(last 4-5 rounds)))
zeta_t = 0.9^4 * r_t^T B_t r_t / (g_t^T B_t g_t + eps)   # estimates the T2 penalty
```

## Honest negative-result verdict
More likely than not that NOTHING cheap beats SGD at short H: SGD removes the
entire transverse-history term; every cheap signal tried estimates the wrong
operator or has the wrong horizon tilt. CTTN is the cleanest lower-bound; a
"clairvoyant recent-span oracle" would upper-bound the cheap-secant class at H16.

## Cross-check
#1 (Chebyshev) is the numerical-analysis stabilized-explicit method for stiff
spectra — expected to converge with the physics/math lit-research (RKC / MD
timestep stability). See docs/LIT_RESEARCH_OPTIMIZATION.md when it lands.
