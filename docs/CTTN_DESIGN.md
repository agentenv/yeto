# CTTN — Curvature-Trust Transverse Nesterov (the oracle test)

The theoretically-correct outer optimizer for the DiLoCo momentum poison, and
the decisive final experiment: an **oracle** (true current-Hessian, directional
damping) that closes the "can any outer optimizer beat memoryless SGD?" question
either way. Design from a Codex (gpt-5.6-sol, xhigh) design pass, 2026-07-13,
after a Codex pre-mortem killed the cheaper directional variant (wsub).

## Why it exists (what everything before got wrong)
- **Momentum memory** poisons short-H (the whole paper). Falsified.
- **Scalar** step/curvature caps (capped-nesterov, curv): anisotropy-blind — one
  `mu` shrinks every direction. Beat SGD only at long H. Falsified.
- **wsub** (directional, worker-disagreement covariance `C=Σ sᵢsᵢᵀ` as an
  H-proxy): the pre-mortem showed `C ≈ α²H²·Σ_worker` is filtered worker-NOISE
  not curvature; the score `E_b` is QUARTIC in delta-scale so it tracks
  magnitude and binds INVERSELY to the poison; rank-3 is blind in LoRA space;
  and `eta=0.147` makes any win a hidden LR effect. Predicted-dead.

CTTN fixes all four: real HVP curvature (not disagreement), per-eigendirection
matrix damping (not scalar `mu`), a dimensionless budget (not the scale-buggy
`E_SUB`), and a parallel step identical to SGD-0.28 (so a win can't be an LR
change).

## The update (per outer round)
Merged pseudo-gradient `g`, incoming Nesterov buffer `b`, momentum `mu`.
```
q = g/||g|| ;  P(v) = v - q(qᵀv)
r = P(b) ;  b_parallel = b - r                 # split buffer into ∥ and ⟂
V,T = block_lanczos(HVP=H_t, Q0=orth([q, r/||r||]), block_steps=4)   # 8 HVPs
Hplus = PSD-project(T)                          # negative curvature -> 0
A = P Hplus P  (in span V) ;  budget = rho * gᵀHplus g   (rho=0.10)
z(tau) = (I + tau A)^{-1} r                     # matrix trust region, ⟂ subspace
tau = smallest tau>=0 with mu⁴ z(tau)ᵀA z(tau) <= budget   # scalar bisection
b_new = mu(b_parallel + z) + g
d = g + mu² z            # == g + mu P(b_new);  qᵀd == ||g|| exactly
theta -= 0.28 * d
```
Sharp Ritz modes of `r` are shrunk `1/(1+tau λ_j)`, flat modes preserved. `rho`
is homogeneous of degree 2 — scaling deltas / inner-LR / loss / eta does not
change whether the cap binds; `tau` is a dual variable, not a hyperparameter.

## Core status — IMPLEMENTED + VALIDATED (`yeto/cttn.py`, `scripts/test_cttn.py`)
The HVP-agnostic dense core is done and passes a golden-trace on a synthetic
Hessian (commit aea1f67):
- `qᵀd == ||g||` exact (parallel step = SGD-0.28);
- `mu⁴ zᵀA z <= rho gᵀHplus g` (trust region binds correctly);
- sharp mode shrunk to 0.078×, flat mode preserved 0.999× (anisotropic);
- dimensionless-`rho` scale invariance (the wsub-killer bug is absent);
- non-bind and degenerate `g=0` fallbacks correct.
Curvature enters ONLY through the `(V,T)` sketch, so the same core runs
wherever the HVPs are produced.

## Cost — research-only (cannot meet the product gate)
8 HVPs/round ≈ 6.3 training-step-equivalents: ~40% compute / ~45% wall overhead
at H16, ~2.5% at H256. Adds a forward pass. Amortizing to <1% needs R>=40 rounds,
which makes the curvature stale. **This is a paper result, not a shippable
default.** SGD-0.28 remains the production optimizer regardless of outcome.

## Will it beat SGD at H16? (the crux — win-win either way)
Codex point-forecast: **yes, narrowly ~0.02** (removes 0.08–0.10 of the Nesterov
penalty) IF useful temporal signal lives in flatter modes than the harmful
displacement. It **cannot** manufacture predictive momentum: if
`E[zₜᵀg_{t+1}] <= 0` after removing sharp modes, the constrained optimum is `z=0`
and CTTN degenerates to SGD — which would be the STRONGEST negative result
("even a current-Hessian directional-damping oracle can't beat SGD short-H →
μ=0 is optimal → the poison is fundamental"). Direct tell: cap binds >90% while
retaining <20% of `||r||` ⇒ μ=0 is the H16 optimum.

## Integration path (from the architecture map)
Outer optimizers run entirely in the Rust syncer on flat deltas — no torch. But
the **action-probe sidecar** (`yeto/action_probe_server.py`) already sits on the
merge path with a torch model + held-out anchor panels + the candidate merged
state crossing the Python↔Rust boundary. CTTN rides that channel:
- `yeto/action_probe_server.py`: enable autograd; compute HVPs (double-backward
  of `sft_loss`) on anchor panels at the eval-point state; block-Lanczos → `(V,T)`;
  run `cttn_step`; return `z` (or `d`).
- `yeto/action_probe.py`: extend request/response schema (carry `g`, `b`, `mu`,
  `rho`, eval-point params → return `z`/`b_new`/diagnostics).
- `syncer/src/action_probe.rs`, `state.rs` (`preview_aggregate_inner`,
  `commit_preview`), `server.rs` (`perform_merge`): ship the merged candidate +
  buffer, receive `z`, apply `d = g + mu² z`, store `b_new`.
- `scripts/compare_diloco.py`: add `cttn` (outer-optimizer + a commit policy to
  trigger the sidecar); wire an anchor manifest as the held-out HVP source.
- Local parity/smoke on a tiny model before GPU (the parity-check gate).

## Pre-registered experiment (24 runs)
- 2 H16 workloads (most- and least-poisoned) × 4 paired seeds × {SGD-0.28, CTTN}.
- On the poisoned H16 workload: a true-HVP **scalar** control `z=αr` at the same
  curvature budget (4 seeds) — isolates anisotropy from mere shrinkage.
- 1 H256 sentinel × 2 paired seeds × {SGD, CTTN}.
- **Success ONLY if:** CTTN > SGD by >0.018 on BOTH H16 workloads, correct sign
  every paired seed, not >0.009 worse on the H256 sentinel, AND beats the scalar-
  HVP control by >=0.009 on the poisoned workload. Else: bank SGD-0.28 and write
  the negative result.

### Instrumentation to log every round (interpretable regardless of outcome)
Rayleigh quotients κ(g),κ(r),κ(z); `E_after/E_before = zᵀĤ₊z/rᵀĤ₊r`; norm
retention `||z||/||r||`; bind rate + per-Ritz shrink factors; # Ritz modes for
90% of `rᵀĤ₊r`; sharp-subspace overlap across rounds; Lanczos residual;
next-round predictive alignment `cos(zₜ, g_{t+1})`; and shadow (non-applied)
SGD-0.28 / fixed-Nesterov-0.9 / scalar-HVP / effective-LR-matched updates.
