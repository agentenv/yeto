# lean-mechanism

Machine-checked (Lean 4 + Mathlib, sorry-free) theorems compressing the
DiLoCo-poison empirical findings onto a minimal anisotropic 2D quadratic model.

- All statements + proofs: `LeanMechanism/Basic.lean`
- Causal phase-locked geodesic safety, exact constant-rotation recovery, and
  reversal non-dominance: `LeanMechanism/CausalPhaseLockedGeodesic.lean`
- Prose summary, proof-status table, counterexample instances, and product
  implication: `../docs/LEAN_THEOREMS.md`

Build: `lake exe cache get && lake build` (toolchain pinned in `lean-toolchain`).

The four results: **T1** the matched-aligned descent gap is exactly the
transverse curvature energy `½ d⊥ᵀH d⊥ + d∥ᵀH d⊥`; **T2** that gap is benign
under isotropic curvature but flips one-step descent into ascent under
anisotropy (exact threshold `ℓ_y⋆ ≈ 33.11` for the concrete instance); **T3** no
geometry-blind scalar step-scale dominates tuned SGD across the quadratic family
(`bestReduction` is proved to be the true scalar loss-optimum); **T4** a
curvature-aware transverse cap restores the one-step descent guarantee.
