# Outer Optimizer Semantics (audit, 2026-07-12)

Authoritative one-page statement of the update the syncer actually applies.
Sources: `syncer/src/merge.rs` (optimizer math), `syncer/src/state.rs`
(aggregate construction, preview/commit), `syncer/src/server.rs` (weighting,
wire boundaries). All merge and optimizer arithmetic is f32 on the syncer.
Deterministic vector tests: `nesterov_three_step_hand_computed_sequence` and
`rho_adaptive_three_step_hand_computed_sequence` in `merge.rs`.

## What δ_t is

Learners push full fragment parameter values θ_m (default wire dtype bf16;
q4 sessions push quantized deltas anchored at the learner's `base_version`,
admitted only when that matches the current version and reconstructed by
adding the current anchor). Per learner, per fragment:

    Δ_m = Θ_p − θ_m        (anchor − learner)

where Θ_p is the **syncer's own current f32 fragment at merge time** — not
the learner's starting point. A stale push (base_version < current) is still
differenced against the current anchor (warn only). Sign convention: Δ is a
pseudo-gradient; with η = 1, μ = 0 the step θ ← θ − ηΔ lands exactly on θ_m.

Learner weights: w_m = c_tokens²/c_steps (`learner_weight`). Merge is per
tensor within a fragment: weighted direct averaging (`merge_avg`) for the
embedding fragment, weighted radial-directional averaging (`merge_rda`) for
all others — merged norm = weighted mean of per-learner delta norms, merged
direction = normalized weighted mean of unit directions (degenerate mean
direction falls back to avg). Optional HeLoCo per-tensor correction of each
learner delta against the fragment momentum buffer runs before merging
(`delta_correction`, default **off**).

**Not present anywhere in this path:** normalization by H or by inner steps
(c_steps enters only the relative weight; a single learner's δ is its raw
parameter displacement whatever H is), gradient clipping, weight decay,
dampening, or LoRA α/r scaling (fragments carry raw adapter A/B tensors; the
peft scaling lives only in the model forward).

## Nesterov (default outer optimizer), per fragment

Buffer b is per fragment, f32, **b_0 = 0** (zeros at construction;
checkpoints persist params + buffer; the rho-adaptive scalar is NOT
persisted). With merged delta δ_t, outer lr η, momentum μ:

    b_t = μ b_{t-1} + δ_t
    d_t = δ_t + μ b_t  =  (1+μ) δ_t + μ² b_{t-1}
    θ_t = θ_{t-1} − η d_t

First commit: d_1 = (1+μ) δ_1, so ‖step_1‖ = (1+μ) η ‖δ_1‖ with zero
history — μ = 0 vs μ = 0.9 at fixed η already scales the current delta by
1.9 before any memory effect exists.

**vs `torch.optim.SGD(momentum=μ, nesterov=True)`:** the recursion is
bit-for-bit the same form at dampening = 0, weight_decay = 0 (torch's
first-step `buf = grad` equals μ·0 + δ). Differences are semantic, not
formulaic: the "gradient" is a merged parameter displacement, not ∇L; math
is f32 on CPU regardless of model dtype; buffers are scoped per flat
fragment (elementwise-identical to per-parameter scoping, but the logged
cosines/ratios are per fragment); dampening/weight-decay are not offered.

## Exact identities (used for logging and the controller)

With c_t = ⟨b_{t-1}, δ_t⟩ / ‖δ_t‖² and r_t = ‖b_{t-1} − c_t δ_t‖ / ‖δ_t‖:

    d_t = A_t δ_t + d_t⊥,   A_t = 1 + μ + μ² c_t,   ‖d_t⊥‖ = μ² r_t ‖δ_t‖

Logged `OuterStepStats`: `applied_step_norm` = η‖d_t‖ (for step-scaled
actions, the post-scale norm); `direction_delta_cosine` = cos(d_t, δ_t);
`history_current_norm_ratio` = μ²‖b_{t-1}‖ / ((1+μ)‖δ_t‖).

## Other outer optimizers

- **normalized-ema:** b_t = β b_{t-1} + (1−β) δ_t, except a zero buffer
  initializes b = δ (unit gain from the first commit); θ −= η b_t.
- **restarted-ema:** same, but when cos(b_{t-1}, δ_t) ≤ threshold the
  history is discarded and b_t = δ_t.
- **rho-adaptive (v2 — v1's μ_eff = clamp(2(1−ρ), 0, μ_max) heuristic was
  retired, docs/EXP2_26.md):** the buffer stores the previously **applied**
  direction s_{t-1} δ_{t-1}. Each commit measures ρ_t = cos(δ_t, b),
  updates ρ̄_t = ½ ρ̄_{t-1} + ½ ρ_t (commits with a zero buffer or zero
  delta leave ρ̄ unchanged; ρ̄_0 = ½), and applies

      s_t = clamp( a(ρ̄_t) / a(½), ½, 2 ),  a(ρ) = 1 + μ*/(1 − μ* ρ), μ* = ½
      θ −= η s_t δ_t;   b ← s_t δ_t

  `--outer-momentum` is not consumed; ρ̄ is per fragment and resets to ½ on
  restore. `history_current_norm_ratio` reports the per-commit ρ_t here.

## Boundaries that shape δ

bf16 wire both ways: pushes quantize θ_m to bf16 before the f32 difference
against the f32 anchor, and broadcasts quantize Θ to bf16, so each learner
restarts its inner phase from a bf16 rounding of the syncer's f32 state.
Action-probe policies (`step_scale`, norm-matched leave-one-out) rescale the
**applied step** after the buffer update — the buffer transition is that of
the unscaled step; params move by s·η d_t on the f32 lattice.
