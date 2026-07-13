# Next-Generation Outer Optimizer: Memoryless Spatial Adaptation

## Decision (2026-07-13, after 3 controller generations failed)

Three controller generations (rho-adaptive v1/v2, capped-nesterov, capped-nesterov-gc/-r)
all failed to beat plain memoryless SGD-0.28 at any horizon. Static-H worst-case
regret: controller ~0.023 vs just-pick-mu=0 ~0.013. Dynamic-H: tuned SGD won,
controller 0.011 behind. Conclusion, treated as settled:

**In this product regime, outer vector first-moment (directional) memory is a
liability. Stop fixing outer momentum. The next optimizer builds on the winning
memoryless SGD base and adds only capabilities that do NOT reintroduce
directional memory.**

STOP doing: v4/v5 momentum caps, more complex rho-controllers, one-step
held-out action selection (killed by the measurement wall), forcing momentum
to be useful to save a paper narrative.

## Two tracks

- **Production default (deploy now):** SGD-0.28, no delta correction — already
  confirmed (EXP2.19) and re-confirmed by every screen since.
- **R&D (next optimizer):** from SGD, add current-round spatial preconditioning,
  safe scalar step adaptation, or model averaging — never a first-moment buffer.

## Target form

```
g = mean(worker_deltas)                    # current round only
u = spatial_precondition(g, worker_deltas) # current round only, no history vec
eta = scalar_step_controller(g, u, tape)   # scalar/second-moment, anchored to 0.28
theta += eta * u
```

Principles: (1) no history direction vector / no outer first-moment; (2) only
improve the current round's merged delta; (3) scalar or per-layer 2nd-moment
stats allowed, no direction-biasing momentum buffer; (4) step size adapts
AROUND the confirmed 0.28, not searched from scratch.

## Candidate bake-off (6 only — no giant grid)

| Candidate | Directional memory | What changes |
|---|---|---|
| SGD-0.28 | none | baseline |
| Iso-C + SGD | none | matrix spectrum of current-round delta (PRIORITY 1) |
| Block RMS, beta1=0 | none | per-tensor/layer scale |
| Block Yogi, beta1=0 | none | robust per-block 2nd-moment scale |
| Worker-SNR SGD | none | same-round cross-worker consensus (most original) |
| best-spatial + scalar-LR tuner | none | direction + global step |

### Priority 1: Iso-C + memoryless SGD
IsoLoCo (arXiv 2607.03011) flattens the merged delta's singular spectrum then
applies SGD — no temporal accumulation, improves current-round spatial
structure, compatible with our "temporal memory is the poison" finding. It
already beat SGD in their DiLoCo experiment (2.929 vs 2.990). Iso-C is
implemented in the syncer (--matrix-merge iso, commit 7b86d7d). First
comparison: SGD-0.28 vs Iso-C+SGD-0.28 across our most sensitive settings.

### Priority 2: second-moment adaptation, no first moment (beta1=0)
Outer AdaGrad/Yogi/RMS with beta1=0 — keep second-moment scaling, drop
direction memory. Use per-tensor/block, NOT per-coordinate Adam:
```
v_{t,l} = beta2 v_{t-1,l} + (1-beta2) ||g_{t,l}||^2 / d_l
u_{t,l} = g_{t,l} / (sqrt(v_{t,l}) + eps)
u_t <- u_t * ||g_t|| / (||u_t|| + eps)   # global norm-match back to SGD
```
First test in isolation: does reallocating update budget across blocks beat SGD,
without also injecting a larger overall step?

### Priority 3: same-round worker disagreement (the likely original contribution)
Free signal a single-worker optimizer lacks: per-block cross-worker consensus.
```
gbar_l = (1/M) sum_i g_{i,l}
sigma_l^2 = (1/(M-1)) sum_i ||g_{i,l} - gbar_l||^2 / d_l
q_l = (||gbar_l||^2/d_l) / (||gbar_l||^2/d_l + sigma_l^2/M + eps)
u_l = q_l * gbar_l   # then global norm-match
```
Keep high-consensus blocks, shrink high-disagreement blocks. No cross-round
memory, no held-out probe -> immune to the one-step measurement wall.
**"Memoryless consensus-aware outer optimization"** — the product differentiator.

### Priority 4: scalar-LR adaptation only (u_t = g_t)
Anchor to SGD-0.28, adapt only eta_t via secant curvature / trust-ratio /
stability-edge / low-freq loss trend. NOT via greedy one-step held-out loss
(known to underperform tuned constant LR long-run). Cap change to 5-10%/round.
```
lambda_hat_t = [<g_t - g_{t-1}, theta_t - theta_{t-1}>]_+ / (||theta_t-theta_{t-1}||^2 + eps)
eta_t = clip(c / (lambda_hat_t + eps), eta_min, eta_max)
```

### Priority 5: primal averaging (last)
Average history MODELS, not gradient directions (Generalized Primal Averaging).
Evidence is single-worker Step-K, not our multi-worker short-H regime -> behind
Iso-C and preconditioning.

## v0 to implement first: Memoryless Block-Adaptive SGD
```
inputs: worker deltas delta_i, base outer LR 0.28
1. g = mean(delta_i)
2. per tensor/block: worker disagreement + current delta RMS
3. block weights: high-consensus large, high-disagreement small
4. global norm-match weighted direction to plain-SGD norm
5. apply with SGD-0.28; NO first-moment momentum
```
Do NOT initially combine with momentum / curvature tuner / Iso-C / primal avg /
dynamic cap. Establish blockwise current-round preconditioning alone first; add
scalar-LR tuner only if it wins.

## Workloads (paired evaluation)
1 production main config; 2 short H; 3 long H; 4 high inner LR; 5 LoRA rank 16;
6 dynamic H; 7 full-parameter control; 8 worker-count / data heterogeneity.
Test matrix must cover the inner optimizers we plan to support (AdamW AND Muon —
MuLoCo shows inner optimizer strongly changes pseudo-gradient quality); do not
pick the outer optimizer on one AdamW config.

## Product success gate
A "better outer optimizer" must satisfy BOTH:
- reaches target loss faster than SGD-0.28, AND
- worst-case regret < SGD-0.28
Concretely: paired improvement > 2x noise floor on >=2 core workloads; never
worse than 1 noise floor on any core workload; NO per-H/rank/inner-LR tuning;
outer-step overhead < ~1% of total training; no extra held-out forward; no spikes
under dynamic H or worker change. (Noise floor ~0.009; a 0.002 win is not a
product improvement.)

## Bake-off order
1. Iso-C + SGD
2. blockwise beta1=0 Yogi/RMS
3. worker-consensus preconditioning
4. best spatial + conservative scalar-LR tuner
5. primal averaging (last)

The likely real differentiator is NOT smarter momentum management but using the
spatial + confidence information the multiple workers provide within a single
round to allocate the update budget better than scalar SGD — with no directional
memory.
