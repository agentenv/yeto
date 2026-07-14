# Exact-state midpoint optimizer preregistration

Freeze date: 2026-07-14 (America/Los_Angeles)
Status: formulas, signs, gates, counterexamples, and two-seed decisions frozen
before any qualifying exact-Adam/H/2 capture exists or is scored.

## Common sign and tensor conventions

For worker `i` and one outer interval:

```text
theta0_i    model parameters at the common outer anchor
thetam_i    parameters after exactly H/2 local optimizer steps
thetaH_i    parameters after exactly H local optimizer steps
a_i         = theta0_i - thetam_i
b_i         = thetam_i - thetaH_i
g_i         = a_i + b_i = theta0_i - thetaH_i
```

`g_i` is the production pseudo-gradient sign: a positive `g` is consumed by
the outer update `theta <- theta - eta_outer * g`. This is the opposite sign
of the learner parameter displacement. All formulas operate independently per
declared production tensor and then use the exact production merge, responder
order, float precision, and worker weights. The final candidate tensor is
grafted to the corresponding factual production-merge norm, so neither rule
silently retunes outer LR 0.28.

For an Adam state snapshot `s in {m,H}`:

```text
mhat_i,s = exp_avg_i,s / (1 - beta1_i ^ step_i,s)
vhat_i,s = metric_numerator_i,s / (1 - beta2_i ^ step_i,s)
P_i,s    = 1 / (sqrt(vhat_i,s) + eps_i)
```

For AMSGrad, `metric_numerator` is the exact `max_exp_avg_sq` used by the
factual update, not ordinary `exp_avg_sq`. Captures must record enough
optimizer implementation metadata to reproduce factual bias correction,
epsilon placement, precision, step convention, scheduler mass, decoupled
weight decay, and any fused/master-weight semantics. A mismatch is an
integrity failure, not a fallback.

`Pbar_s = weighted_mean_i(P_i,s)` is the common inverse-RMS metric. `RDA(S)`
below means exact production per-tensor RDA on worker vectors `S`, including
its factual degeneracy fallback.

## Candidate 1: MTRF — Metric-Transported Richardson Force

MTRF asks whether the two half-interval displacements represent a changing
underlying force after removing Adam's changing diagonal metric and scheduler
mass.

Let `L1_i` and `L2_i` be the exact sums of factual local learning rates over
the first and second halves, excluding decoupled weight-decay movement. For
each worker and coordinate:

```text
f1_i = a_i / (L1_i * P_i,m)
f2_i = b_i / (L2_i * P_i,H)
r_i  = 2*f2_i - f1_i
h_i  = 2*mhat_i,H - mhat_i,m
z_i  = Pbar_H * r_i
q_i  = Pbar_H * h_i
```

`r_i` is midpoint Richardson extrapolation in approximate unpreconditioned
force space; `z_i` transports it through the common endpoint metric. On each
tensor, use `z_i` only if `dot(z_i,q_i) > 0` and all required values/norms are
valid; otherwise use factual `g_i`. Graft the accepted worker tensor to
`||g_i||`, production-merge the four worker candidates, and finally graft the
merged tensor to the factual `||RDA(g_i)||`.

This rule has no fitted coefficient, decay, cosine margin, LR multiplier, or
horizon-specific constant. The strict sign gate checks agreement between the
displacement-derived force trend and the independently captured Adam first
moment trend.

Expected fixed point: when both halves have the same force and metric and the
Adam moment agrees, Richardson returns the same force direction; after norm
grafts MTRF is factual baseline. It acts only on measurable half-path/metric
evolution.

### MTRF counterexamples fixed in advance

- Adam first moments can lag a genuinely rotating gradient, so the sign gate
  may reject a useful turn or accept a stale one.
- Dividing a multi-step displacement by a terminal diagonal metric is not an
  exact recovery of every hidden per-step gradient; the formula is fully
  identifiable but remains a model of the path.
- Strong AdamW decay, clipping, dynamic loss scaling, or skipped steps make
  raw displacement an invalid force proxy unless their exact contributions
  are separately captured and removed.
- Coordinatewise Adam metrics are not invariant to arbitrary LoRA gauge
  transformations. Gauge changes between snapshots invalidate the record.
- A force can change abruptly after H even when Richardson perfectly explains
  the observed two halves; therefore finite-loss and next-update gates are
  mandatory.

## Candidate 2: MSTP — Moment-Secant Trust Projection

MSTP uses the exact bias-corrected Adam direction directly but bounds its
departure from production by the actually observed turn between the two
half-paths.

For each tensor, first merge factual halves independently:

```text
A = RDA({a_i})
B = RDA({b_i})
G = RDA({g_i})              # factual full-interval baseline
Z = 2*weighted_mean_i(P_i,H * mhat_i,H)
    - weighted_mean_i(P_i,m * mhat_i,m)
Q = graft(Z, ||G||)
Delta = Q - G
R = min(||G||, ||B - A||)
alpha = 0                         if ||Delta|| = 0
        min(1, R / ||Delta||)     otherwise
Craw = G + alpha*Delta
C = graft(Craw, ||G||)
```

`Z` is a metric-aware secant extrapolation of the exact Adam update direction.
The trust radius is parameter-free: it cannot exceed the baseline norm and is
also limited by the observed difference between first- and second-half merged
paths. If the path is stationary (`A=B`) the radius is zero and MSTP is
exactly baseline. No worker is held out and no same-round prediction target
selects the action.

### MSTP counterexamples fixed in advance

- Adam moment lag can make `Z` point behind a fast rotation; the trust ball
  bounds but cannot prove benefit.
- A small `||B-A||` can hide large coordinatewise rotations through
  cancellation, causing over-conservative abstention.
- A large half-path turn may be stochastic noise, granting too much radius to
  a harmful moment extrapolation.
- Metric/moment state excludes decoupled weight decay, clipping history, and
  data-order effects unless explicitly captured.
- Norm graft controls scale but does not control loss curvature; even a small
  angular action can increase finite loss.

## Causal targets

### Primary: common-random-number finite-loss microfork

At each frozen boundary, restore identical model parameters, buffers, exact
learner Adam states, schedulers, AMP/scaler state, RNG state, and the next
predeclared batch-group IDs. Apply exactly one factual outer SGD-0.28 action
using baseline, MTRF, or MSTP. Evaluate all three on the same fixed evaluation
microbatch immediately (`k=0`), then run exactly eight identical local update
groups per worker and evaluate again (`k=8`). No arm may update buffers during
evaluation. Arm order is deterministically rotated by boundary ID.

Per boundary and horizon:

```text
gain_candidate = NLL_baseline - NLL_candidate
```

Positive is better. The candidate formula and action must be sealed before
any arm loss is computed. Inference pairs arms within boundary.

### Secondary: sealed next-same-fragment direction

Seal factual and candidate directions at group `t`; when the next group for
the same fragment arrives, score

```text
cos(candidate_t, G_t+1) - cos(G_t, G_t+1).
```

The future direction cannot construct, tune, gate, or renormalize its own
candidate. This is a mechanism diagnostic, never a substitute for finite loss.

## Frozen sample, safety, and promotion gates

Development seed 223 must satisfy every candidate-specific gate independently:

- at least 32 complete CRN boundaries, balanced at least eight per fragment;
- exact factual AdamW replay and production merge/outer update at every
  boundary, maximum parameter error `<=1e-6` and no state mismatch;
- no missing/non-finite fields; midpoint is exactly H/2 accepted local steps
  for every worker, not wall-clock or attempted-step midpoint;
- mean paired `k=8` NLL gain strictly greater than `0.002`;
- 95% fragment-stratified boundary-bootstrap lower endpoint for `k=8` gain
  greater than zero;
- at least 3/4 fragment `k=8` means positive and at least 60% of individual
  boundaries positive;
- `k=0` mean gain not below `-0.001`, with 5th-percentile paired gain above
  `-0.01`;
- mean sealed next-direction cosine gain greater than `0.001`, moving-block
  95% lower endpoint greater than zero, and at least 3/4 fragments positive;
- candidate acts on at least 25% of boundaries/tensors, with conditional
  median full-action relative norm at least 2%;
- full-action p99 angle below 20 degrees and maximum below 30 degrees;
- no candidate boundary has NLL increase greater than `0.05` at either
  horizon; candidate/factual runtime overhead below 2%;
- tensor fallback at most 1% (excluding specified stationary MSTP zero-radius
  abstention, which is reported as no-action rather than fallback).

The bootstrap uses 20,000 deterministic replicates, seed 5318008. Boundaries
are clusters and resampling is stratified by fragment. MTRF and MSTP are two
simultaneous hypotheses: development lower confidence endpoints use Holm
step-down correction at familywise alpha 0.05. Candidate ranking is frozen as:

1. highest Holm-corrected `k=8` lower endpoint;
2. then highest mean `k=8` gain;
3. then lower p99 action angle;
4. then lexical name (`MSTP` before `MTRF`) for an exact tie.

## Two-seed decision rule

1. Run seed 223 development only after schema/integrity checks pass.
2. If neither candidate passes every corrected development gate: **KILL BOTH**.
3. If one or both pass, write and hash a selection manifest containing the
   frozen winner, formulas, executable/config hashes, seed-239 commitment,
   and all development outcomes. Non-winners are killed.
4. Only after that manifest is immutable, open seed 239 and rerun the selected
   candidate with the identical formula, gates, CRN schedule shape, and
   uncorrected single-hypothesis alpha 0.05 confidence interval.
5. Seed 239 must independently pass every gate. A failure is **KILL** and seed
   223 cannot rescue it.
6. A two-seed pass is **PROCEED TO H16/H256 AND MODEL/INNER-OPTIMIZER BREADTH**,
   not permission to replace production SGD-0.28. At least one additional
   model and both SGD/AdamW inner optimizers must pass before an online claim.

No threshold, coefficient, action gate, target horizon, sample exclusion, or
candidate ranking may change after seed 223 outcomes are visible.

## Exact capture field requirements

Per boundary, worker, parameter group, tensor, and snapshot as applicable:

- immutable boundary ID, seed, worker ID, fragment ID/version, responder order,
  base version, H, accepted local-step count, token count, worker weight;
- exact `theta0`, `thetam`, `thetaH` trainable parameters in optimizer/master
  precision, plus buffers needed by forward evaluation;
- exact Adam `exp_avg`, `exp_avg_sq`, AMSGrad `max_exp_avg_sq` if enabled,
  optimizer step, beta1/beta2, epsilon and placement, maximize/capturable/
  differentiable/fused/foreach flags, master-weight and stochastic-rounding
  behavior at midpoint and endpoint; start state is also required for factual
  replay validation;
- exact per-step LR schedule or first/second-half LR sums, weight decay and
  its parameter-group mapping, clipping coefficients/norms, skipped-step and
  gradient-scaler history, and an explicit decomposition/removal of decoupled
  weight-decay displacement for MTRF;
- exact tensor boundaries, dtype, merge mode, production worker weights,
  accumulation/responder order, outer LR and optimizer state;
- full model/buffer state, learner optimizer/scheduler/scaler state, CPU/CUDA
  RNG state, data sampler state, and immutable IDs/hashes for the next eight
  training batch groups and fixed evaluation microbatch;
- sealed factual/MTRF/MSTP direction hashes before evaluation, immediate and
  k=8 per-arm finite losses, timing, and all failure/overflow events;
- next same-fragment factual production direction for secondary scoring;
- source commit, image, model, data, split, command, and analysis-config hashes.

Any missing coordinate state, silent dtype cast, unknown decay contribution,
or H/2 ambiguity makes the candidate **UNIDENTIFIABLE**, not zero-action.
