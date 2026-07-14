# Exact-Adam/H/2 optimizer design report

Date: 2026-07-14
Outcome: **two preregistered candidates; both currently UNIDENTIFIABLE**

## Ranked candidates

### 1. MSTP — Moment-Secant Trust Projection

MSTP is the stronger first capture target. It forms a metric-aware secant
extrapolation from exact bias-corrected Adam directions at H/2 and H, then
clips departure from factual production RDA by the observed difference between
the first- and second-half merged paths. The trust radius is
`min(||G||, ||B-A||)` and the final tensor is grafted to `||G||`.

Why ranked first:

- the midpoint supplies a directly observable, parameter-free trust radius;
- exact Adam moments and metrics supply the proposed direction rather than
  attempting to recover every hidden gradient from displacement;
- stationary halves imply an exact zero-radius baseline action;
- action magnitude has a narrow algebraic bound and no tuned coefficient;
- it needs less decay-path modeling than MTRF.

Main risk: Adam moment lag or noisy half-path turns can still authorize the
wrong rotation. The trust ball bounds geometry, not finite loss.

### 2. MTRF — Metric-Transported Richardson Force

MTRF divides each half displacement by its exact LR mass and Adam inverse-RMS
metric, Richardson-extrapolates the resulting approximate force, remaps it
through a cross-worker endpoint metric, and accepts a worker tensor only when
the displacement-derived and exact-first-moment trends have positive dot
product. Worker and merged tensors are norm-grafted to factual scale.

Why ranked second:

- it is the more direct test of the midpoint/metric-confounding hypothesis and
  may detect useful force evolution invisible to endpoint-only methods;
- but a multi-step displacement divided by a terminal half metric is an
  identifiable model, not exact recovery of hidden per-step gradients;
- it requires exact LR masses and verified removal of AdamW decay, clipping,
  skipped-step, and scaler effects;
- it is more coordinate/gauge sensitive and has the larger capture burden.

The prior plausibility ranking above does not override the frozen empirical
selection rule: among candidates that pass every seed-223 gate, selection is
by highest Holm-corrected k=8 finite-loss lower endpoint, then mean gain, p99
angle, and finally lexical tie-break.

## Formula and sign summary

Production sign is fixed throughout:

```text
a_i = theta0_i - thetam_i
b_i = thetam_i - thetaH_i
g_i = a_i + b_i = theta0_i - thetaH_i
theta_outer <- theta_outer - 0.28 * merge(g_i)
```

For exact bias-corrected Adam state:

```text
mhat_i,s = exp_avg_i,s / (1-beta1^step_i,s)
vhat_i,s = factual_metric_state_i,s / (1-beta2^step_i,s)
P_i,s    = 1 / (sqrt(vhat_i,s)+eps)
```

MTRF:

```text
f1_i = a_i / (L1_i*P_i,m)
f2_i = b_i / (L2_i*P_i,H)
r_i  = 2*f2_i-f1_i
z_i  = Pbar_H*r_i
q_i  = Pbar_H*(2*mhat_i,H-mhat_i,m)
```

Use norm-matched `z_i` only when `dot(z_i,q_i)>0`; otherwise use factual
`g_i`, then exact production merge and a final factual-norm graft.

MSTP:

```text
A = RDA(a_i); B = RDA(b_i); G = RDA(g_i)
Z = 2*mean_w(P_i,H*mhat_i,H) - mean_w(P_i,m*mhat_i,m)
Q = graft(Z, ||G||)
Delta = Q-G
R = min(||G||, ||B-A||)
alpha = 0 if ||Delta||=0 else min(1,R/||Delta||)
C = graft(G+alpha*Delta, ||G||)
```

Full formulas, specified fallbacks, counterexamples, causal target construction,
Holm correction, gates, and the two-seed decision are frozen in `PREREG.md`.

## Decision and target protocol

The primary target is a common-random-number finite-loss microfork at at least
32 balanced boundaries. Baseline, MTRF, and MSTP start from identical model,
buffer, exact optimizer/scheduler/scaler, RNG, sampler, and batch state. Each
receives exactly one outer SGD-0.28 direction. Paired fixed-microbatch NLL is
measured immediately and after eight identical local update groups.

The secondary target seals each direction before scoring it against the next
same-fragment factual production direction. Same-round held-worker prediction
is deliberately excluded because IQM showed it can favor a direction that is
uniformly harmful temporally.

Seed 223 development requires, independently for each candidate:

- exact factual optimizer/merge/outer replay;
- at least 32 boundaries, at least eight per fragment;
- k=8 paired NLL mean gain >0.002, corrected lower confidence endpoint >0,
  3/4 positive fragment means, and at least 60% positive boundaries;
- non-harmful immediate loss and bounded tails;
- next-direction mean gain >0.001, lower endpoint >0, 3/4 positive fragments;
- nontrivial action, p99 angle <20 degrees, maximum <30 degrees;
- no loss spike >0.05, fallback <=1%, and runtime overhead <2%.

The two candidate intervals use Holm familywise correction. Only a seed-223
winner frozen in a hashed selection manifest may open seed 239. Seed 239 must
independently pass the unchanged single-candidate gates. A two-seed pass only
promotes to H16/H256, another model, and SGD/AdamW-inner breadth; it does not
replace production SGD-0.28.

## Exact capture requirements

The recorder must provide, per boundary:

- IDs/provenance: boundary, seed, fragment/version, worker IDs, responder
  order, base version, exact H, accepted H/2 and H step counts, tokens, weights,
  source/image/model/data/config hashes;
- parameters: exact `theta0`, `thetam`, `thetaH` in optimizer/master precision,
  plus all forward buffers;
- Adam state: `exp_avg`, exact factual metric state (`exp_avg_sq` or AMSGrad
  `max_exp_avg_sq`), optimizer step, beta values, epsilon and placement,
  parameter groups, flags, dtype, master weights, fused/foreach behavior at
  start, midpoint, and endpoint;
- path accounting: exact per-step LR or half LR sums, decoupled weight-decay
  decomposition, clipping coefficients/norms, skipped-step/scaler history,
  stochastic-rounding and loss-scaling state;
- merge: tensor bounds/names, dtype, merge mode, weights and accumulation order,
  factual outer direction/state/LR;
- CRN: full model/buffer and learner optimizer restore state, CPU/CUDA RNG,
  sampler state, immutable next-eight batch IDs/hashes, fixed evaluation batch,
  deterministic arm order, and pre-evaluation action hashes;
- outcomes: per-arm immediate/k=8 losses, timing, overflow/failure events, and
  next same-fragment factual direction.

`CAPTURE_SCHEMA.md` supplies the concrete index and NPZ contract. Unknown
decay contribution, ambiguous midpoint, missing metric coordinate, or silent
precision conversion yields **UNIDENTIFIABLE**, never a zero-action result.

## Current retained-artifact audit

The existing seed-223 `syncer_probe_capture_v1` index was checked read-only.
All 320 rows lack the exact midpoint/Adam/CRN bundle references and campaign
provenance fields required by the new schema. The machine-readable audit is
`current_capture_audit.json` and correctly returns:

```text
decision: UNIDENTIFIABLE
identifiable: false
rows with missing fields: 320/320
```

No current vector outcome was scored. No live experiment or cloud state was
accessed.

## CPU replay skeleton

`exact_state_midpoint_replay.py`:

- rejects missing exact-state arrays;
- validates worker/state shapes and tensor bounds;
- reconstructs production per-tensor RDA and reports maximum baseline error;
- bias-corrects exact Adam moment/metric state;
- computes MTRF and MSTP with frozen signs, gates, trust radius, and norm grafts;
- hashes candidate direction bytes before optional targets;
- scores optional sealed next-direction and CRN k=0/k=8 losses;
- audits existing JSONL indexes for identification fields;
- contains a stationary-path self-test in which both candidates reduce to
  factual baseline.

Commands:

```bash
.venv/bin/python /tmp/optimizer_seventh_round2/exact_state_midpoint_replay.py --self-test

.venv/bin/python /tmp/optimizer_seventh_round2/exact_state_midpoint_replay.py \
  --audit-index PATH/index.jsonl --out audit.json

.venv/bin/python /tmp/optimizer_seventh_round2/exact_state_midpoint_replay.py \
  --record boundary0.npz --record boundary1.npz --out replay.json

cd lean-mechanism
lake env lean /tmp/optimizer_seventh_round2/Mechanism.lean
```

The replay skeleton is intentionally not a production-exact AdamW engine. The
capture campaign must first prove exact factual optimizer-state transitions;
the skeleton then evaluates candidate geometry and sealed outcomes.

## Lean scope

`Mechanism.lean` proves without `sorry`:

- stationary-force Richardson extrapolation is a fixed point;
- interpolation with a coefficient in `[0,1]` cannot exceed the proposed
  correction norm;
- equal half-path merges give MSTP zero turn radius.

The direct Lean check passes. These are narrow mechanism identities only, not
stochastic convergence or loss-improvement theorems.

## Artifact hashes

```text
PREREG.md                       2bee9a062ed09c0151a2371337a282ac137635f50f76915e0d4a6f275257a24b
CAPTURE_SCHEMA.md                518d2f90414b794ab2ea953c0d17041cb1afda1261fbc4a611677193eff0506b
exact_state_midpoint_replay.py   19f82d4c4c90ad4d006a77fbe407a456140cb7cccfe4f0f70d364da04b4db101
Mechanism.lean                   cb2ac5bad2ec4441d469e5194e44a8738574d0b483c972e9619b68417582aa67
current_capture_audit.json       918ca265540528f59c8c1a336d61cf3d19443d6dfeaf2507f1320de474df1008
```
