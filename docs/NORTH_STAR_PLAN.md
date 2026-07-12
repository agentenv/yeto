# North-Star Paper Plan (advisor synthesis, 2026-07-12)

Central claim: **temporal correlation is the missing state variable governing
outer momentum in two-phase distributed optimization.** H, staleness, worker
count, and training stage matter largely through the geometry of the
pseudo-gradient sequence. Measuring that geometry enables a
zero-evaluation-cost controller that stays near-optimal as the regime changes.

Do NOT center on "momentum is poison" / "staleness innocent" / "smart merging
impossible". Contributions list (only three): (1) mechanism — outer Nesterov
is a temporal filter whose data-dependent gain and transverse accumulation are
set by the pseudo-gradient correlation kernel; (2) evidence — controlled
interventions + cross-regime collapse predict the short-horizon phase
transition better than H or staleness; (3) method + consequence — a
zero-eval-cost gain controller transfers without retuning and enables elastic
synchronization.

Novelty pressure: 2509.10439 (outer LR must be tuned with sync frequency;
Nesterov instability known), SNOO (pseudo-gradient momentum beneficial at
scale), 2605.28585 (momentum restarting), HeLoCo 2606.00271 (cosine-vs-momentum
correction), **IsoLoCo 2607.03011 (Jul 3 2026: merge transform beats momentum
DiLoCo — wall claim must narrow to validation-guided selection under
affordable probe budgets)**.

## Exact mechanism (Phase 1-2)

Our update: b_t = μb_{t-1} + δ_t; d_t = δ_t + μb_t = (1+μ)δ_t + μ²b_{t-1}.
So μ=0.9 multiplies the CURRENT delta by 1.9 with zero history — μ=0 vs 0.9 at
fixed η does not isolate memory. Exact identities (no AR(1) needed):
c_t = <b_{t-1},g_t>/|g_t|²; r_t = |b_{t-1} − c_t g_t|/|g_t|;
d_t = A_t g_t + d_t⊥ with A_t = 1+μ+μ²c_t and |d_t⊥|/|g_t| = μ²r_t.
E[A_t] = 1 + μ + μ² Σ_k μ^{k-1}ρ_k → 1 + μ/(1−μρ) for geometric ρ_k.
Log the full lag kernel ρ_k, A_t, r_t on every run.

Required: one-page optimizer-semantics appendix + deterministic vector unit
tests (sign, buffer init, Nesterov form, dampening, δ normalization, BF16,
LoRA scaling).

Phase map: H×μ×η on a small FULL-PARAMETER model (use in-repo SmolLM2-135M
infra), H∈{8..512}, μ∈{0,.3,.5,.7,.9,.95}, log-η grid, ≥3 paired seeds near
the boundary, sequential design (early-terminate divergences). Mandatory
fairness controls per H: fixed nominal LR; current-gradient-matched (÷(1+μ));
observed-aligned-gain-matched; full update-norm matched; independently tuned
SGD and Nesterov. GATE: proceed only if conventional LR matching does NOT
remove the effect while correlation-aware matching collapses it.

## Causality (Phase 3)

Buffer-orientation intervention: same checkpoint, same merged delta, momentum
buffers with same norm but different orientation (real / aligned / orthogonal
/ anti-aligned / random-rotated); one outer step; high-precision paired loss.
Manipulate ρ independent of H (training stage, data overlap, inner LR at
matched delta norm, LoRA rank, buffer reset/transport, δ-mixture); pairs with
different H but matched ρ or A_t. Separate temporal cos(ḡ_t, ḡ_{t-1}) from
worker agreement mean_i cos(g_t^i, ḡ_t); test where IsoLoCo's Iso-C acts.

## Theory (Phase 4)

A: exact filter decomposition (varying μ_t, nonstationary norms, finite
history). B: correlated-input quadratic analysis — aligned amplification,
transverse energy, stability region in (η,μ,ρ-kernel), regime where standard
LR adjustment insufficient but correlation-aware sufficient. C: safety
guarantee — controller caps aligned gain + transverse accumulation, preserving
one-step descent whenever memoryless SGD would.

## Controller (Phase 5)

Candidate 1 (preferred): gain-normalized Nesterov — s_t = min(1, (1+μ)/A_t);
θ ← θ − η s_t d_t. Exactly Nesterov at design point; never enlarges; zero new
hyperparameters. Candidate 2: largest μ_t with 1+μ_t+μ_t²[c_t]_+ ≤ 1+μ_max
(optionally cap μ_t²r_t ≤ τ⊥). Candidate 3: separate ∥/⊥ control (keep only
if 1 underperforms). Develop on one small full-param + one LoRA setup, freeze
formula/constants/warmup/clipping, then never modify. Primary metric:
worst-case regret vs best per-setting tuned fixed optimizer; catastrophic-cell
count; tuning-FLOPs saved.

## Confirmatory transfer (Phase 6)

Frozen eval: full-param ~150M/1B/larger; LoRA multiple ranks incl. 9B;
full-param post-training; M∈{1,4,16}; IID + non-IID; AdamW + Muon inner; H
short→500; early/mid/late; sync + controlled-delay + heterogeneous async;
dynamic H. Baselines: tuned SGD/heavy-ball/Nesterov/Schedule-Free; per-H tuned
Nesterov; clipping/norm/trust-ratio/momentum-reset; Delayed Nesterov;
restarting; HeLoCo (sync ablation); IsoLoCo; SNOO (M=1). Must preserve the
good long-H regime, not just protect short-H LoRA.

## Systems payoff (Phase 7)

Dynamic H under bandwidth variation / worker churn / failure recovery: frozen
controller vs best fixed policies vs per-H tuned oracle. Metrics: wall-clock
and FLOPs to target loss, comm volume, recovery, failure probability.
Staleness in optimization units (version lag, parameter displacement, buffer
displacement, arrival cosine), crossed with ρ and gain. Expected: moderate
staleness is not primary; amplification exists synchronously.

## Measurement wall (Phase 8, secondary/companion)

First validate harmful-merge labels (large panels, repeated paired eval,
common random numbers, CIs → latent true rate). Lower bound
n = Ω((σ²_pair/Δ²)log(K/δ)) translated into eval tokens/FLOPs/net gain; panel
curve 1..128; baselines: paired eval, sequential elimination, LUCB, SPRT,
first-order estimates. Correct claim: affordable held-out evaluation is
information-limited for closely spaced merge actions after early training.

## Six figures

1 reversal phase diagram (tuned controls); 2 exact mechanism (A_t, r_t vs
observed); 3 causal buffer intervention; 4 universal collapse vs
correlation-adjusted gain (χ_t = ηA_t·λ̂ curvature-normalized variant);
5 frozen controller worst-case regret; 6 dynamic-H systems win. Wall = Fig 7
or appendix.

## Immediate run order

1 semantics audit; 2 SmolLM2 H×μ×η phase map; 3 LR-matching controls;
4 log lag kernel + A_t + r_t; 5 buffer-orientation intervention; 6 same-H
diff-ρ and diff-H matched-ρ; 7 gain-normalization controller vs clipping vs
tuned Nesterov; 8 freeze; 9 held-out transfer; 10 dynamic-H experiment;
11 restart/HeLoCo/IsoLoCo/Schedule-Free baselines; 12 large confirmatory run.

## Kill criteria

Standard LR matching removes effect → mechanism not novel. ρ/A_t/r⊥ fail
held-out prediction → no universal state variable. Buffer intervention fails →
epiphenomenon. Simple clip matches controller → theory must carry. Needs
per-setting tuning → drop "tuning-free". Sacrifices long-H benefit → safety
patch only. No systems win → optimization paper, not best-paper. Wall bound
dies under sequential testing → failed probe implementation, not
impossibility.
