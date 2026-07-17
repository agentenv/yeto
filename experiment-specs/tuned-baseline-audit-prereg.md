# Tuned-Baseline Audit Preregistration

**Registered thesis:** *A tuned-baseline audit of outer optimization for communication-efficient training*

**Revision:** audit pivot v1.2 ceiling correction and launch-surface cap, 2026-07-17

**Status:** prospective for A1, A2, A3, A4, A5, A6(i), and A6(iii); A6(ii) is an explicitly outcome-known citation of a sealed historical result and contributes no new confirmatory seed.

**Machine-readable companion:** `experiment-specs/tuned-baseline-audit-prereg.json`

## 0. Supersession, preserved history, and scope

This document supersedes the prior prospective mechanism/controller program in `EXPERIMENT-PROGRAM.md`, whose exact pre-pivot bytes are preserved at `history/EXPERIMENT-PROGRAM-pre-audit-2026-07-16.md` with SHA-256 `7ece64ac0c828b183266c451b61c196dbab89762d95703bcf820ca01f7f8e958`.

### 0.1 Ceiling-amendment revision history

Revision 1.2 corrects A1's hard ceiling after the operator established that revision 1.1's `$82.93` cap was below A1's approximately `$106` cheapest complete one-GPU path. It also caps A1/A3/A4 at two concurrent atomic blocks and adds a separate `$40` per-stage pre-science abort-burn kill. It does not change a scientific cell, seed, command, outcome rule, gate, retry rule, blinding rule, or the `$2,485` program hard ceiling.

| Revision | Status | Exact source identity | A1 ceiling | A3 ceiling | A4 ceiling | Width cap | Abort-burn kill |
|---|---|---|---:|---:|---:|---:|---:|
| v1.0 | superseded for remaining-stage cost authority | Git `c40d525ee509d566c7dd7c9c6e94fa4083a270a5`; JSON SHA-256 `a2eb92161a324f161fcdb6d808ae0a9e2309a4bb0dd3924d78122d4d1acad62a`; Markdown SHA-256 `7758ff4a54627bde8418bdd26b93f0058bf7ff917e22805155274f9169b187f0` | `$75.00` | `$40.00` | `$140.00` | none | none |
| v1.1 | superseded by the ceiling correction | Git `e35d82143c3a50c3c1a6ef7327fb40e47bb0ff57`; JSON SHA-256 `bf66f7b07fb3b5a8ed531320e4ca505df51f75d4d6fd22aa4ab6edc3eafe7bed`; Markdown SHA-256 `f0fec5d8e2b479136c473a1123acc42047f598aeb8839701d25f994ed7ddcae3` | `$82.93` | `$31.18` | `$138.21` | 5 blocks | none |
| v1.2 | current ceiling correction | operator authorization 2, 2026-07-17 | `$140.00` | `$31.18` | `$138.21` | 2 blocks for A1/A3/A4 | `$40.00` per stage |

The A1 increase is funded without increasing the program total. Raising A1 from `$82.93` to `$140.00` requires `$57.07`: `$2.68` comes from the previously unallocated program remainder and `$54.39` is drawn from A6(i) OMR's block-ceiling reserve, reducing that ceiling from `$1,000.00` to `$945.61`. OMR's registered planning maximum remains `$800`. The new sum of all block ceilings is exactly `$2,485.00`, so it fits the unchanged `$2,485` program hard ceiling with `$0.00` unallocated.

The `$9.231503` spent on the two retired pre-science A1 launch families remains preserved in the operational history but is excluded from the fresh amended science budgets by explicit operator authorization. It purchased only the subsequently landed transient-provider isolation and fail-fast cost-guard fixes. It does not authorize deletion or mutation of any retired prefix.

The supersession is scientific, not archival:

- `P1R0-FINAL.md`, `P1-ADAPTIVE-FINAL.md`, `PAPER-9B.md`, the frozen phase-map preregistration, and all sealed acquisition/replay artifacts remain unchanged.
- The failed P1 tuned-LR gate is retained as the result that forced the pivot. It is development evidence, not fresh confirmation.
- The prior “momentum poison,” helpful-to-harmful phase transition, universal temporal-state, and controller-payoff headlines are no longer prospective targets.
- The exact temporal-filter algebra remains useful, but the first-order learning-rate equivalence is treated as classical baseline hygiene. The residual scientific target is the tuned loss-versus-communication frontier and the second-order term that may remain after tuning.
- Any earlier deflation framing is superseded by this audit framing. Negative, null, surviving, reversed, and collapsed effects are all publishable outcomes under the classifications below.

Experiment-label history is explicit:

| Current label | Prior label | Disposition |
|---|---|---|
| A1 | E1 | Mechanics retained; gate reframed as seed-level tuned collapse |
| A2 | E2 | Mechanics retained; now the cross-stack audit row |
| A3 | new | Tuned `mu=0` loss-versus-communication frontier and kernel-law prediction |
| A4 | E4 | M-axis mechanics retained |
| A5 | E5 | Buffer surgery retained; now isolates the second-order term after scale/tuning controls |
| A6 | new | Literature audit reproductions: OMR, sealed SCAFFOLD-lite deconfound, and SlowMo |

## 1. Audit thesis and estimands

The paper-level hypothesis is:

> Many reported outer-optimizer gains or failures in DiLoCo-style training are comparisons against a fixed outer learning rate. Once every arm receives an independently bracketed learning rate and training seeds are paired, the apparent method effect will often attenuate or collapse. The corrected empirical object is the tuned loss-versus-communication frontier. A finite-history temporal-filter law should predict that frontier from the measured pseudo-gradient kernel, while curvature-weighted buffer energy is the principal term that can survive first-order tuning.

This is a hypothesis, not an outcome constraint. Each audit row first asks whether the published/fixed-eta effect replicates and then whether it survives independent per-arm tuning.

For method `a`, horizon `H`, seed `s`, and outer learning rate `eta`, define

```text
L(a,H,eta,s)       = locked endpoint audit loss
eta_star(a,H)      = development-selected, interior, independently bracketed eta
T(a,H,s)           = L(a,H,eta_star(a,H),s)
Delta_fixed(H,s)   = L(method,H,eta_pub,s) - L(baseline,H,eta_pub,s)
Delta_tuned(H,s)   = T(method,H,s) - T(baseline,H,s)
F(H)               = T(mu=0,H), the tuned memoryless frontier
```

Negative deltas favor the candidate method. `eta_pub` is the common fixed learning rate used by the original claim or its exact published recipe. The tuning budget, grid cardinality, boundary-extension rule, evaluation set, and seed set must be identical across the method and baseline within an audit row.

### Outcome classification

Every prospective audit contrast receives exactly one label:

| Label | Frozen rule |
|---|---|
| `FIXED_NOT_REPLICATED` | Fixed-eta effect has the wrong sign or its Holm-adjusted seed-level 95% CI includes zero |
| `COLLAPSES_WITH_TUNING` | Fixed effect replicates and the tuned-effect CI lies wholly inside `[-epsilon,+epsilon]` |
| `SURVIVES_TUNING` | Fixed effect replicates; the tuned-effect CI excludes zero in the published direction; and `abs(mean Delta_tuned) >= epsilon` |
| `REVERSES_WITH_TUNING` | Fixed effect replicates but the tuned-effect CI excludes zero in the opposite direction |
| `ATTENUATED_OR_INCONCLUSIVE` | None of the preceding rules is met |

For SmolLM2-135M, `epsilon=0.010` NLL. For the Qwen3.5-9B LoRA row, `epsilon=0.009` NLL; this is a prospective practical-equivalence margin inherited from that stack, not a claim that `0.009` is a universal noise floor. For OMR, `epsilon` is the validation-NLL improvement produced by 5% additional tokens on a separately frozen baseline compute curve. That scalar must be sealed before any OMR method outcome is evaluated. For SlowMo/CIFAR-10, `epsilon=0.25` validation-accuracy percentage points.

## 2. Verification of the first-order learning-rate law

The P1 adaptive report’s winning-eta column permits a direct check of

```text
eta_star(mu) = (1-mu) eta_star(0).
```

The fit below uses only the six positive-momentum winners; the three `mu=0` rows are not counted as automatic exact fits.

| H | mu | predicted eta | observed winning eta | observed / predicted |
|---:|---:|---:|---:|---:|
| 16 | 0.5 | 0.0109375 | 0.0109375 | 1.000 |
| 16 | 0.9 | 0.0021875 | 0.002734375 | 1.250 |
| 64 | 0.5 | 0.0109375 | 0.0109375 | 1.000 |
| 64 | 0.9 | 0.0021875 | 0.002734375 | 1.250 |
| 256 | 0.5 | 0.021875 | 0.021875 | 1.000 |
| 256 | 0.9 | 0.004375 | 0.0109375 | 2.500 |

Fit summaries:

- Regressing observed eta on the law prediction through the origin gives slope `1.0416667`, `R^2=0.8300`, and RMSE `0.0026573`.
- Holding the coefficient at the literal identity gives `R^2=0.8248`, RMSE `0.0026977`, and MAE `0.0012760`.
- On the normalized ratios `eta_star(mu)/eta_star(0)` versus `1-mu`, the through-origin slope is `1.0256410`, `R^2=0.8687783`, and RMSE `0.0622323` in ratio units.
- Three of six positive-momentum winners are exact; five of six are within 25% of the prediction. The median absolute relative error is 12.5%.
- Excluding the single `(H=256,mu=.9)` leverage point, the through-origin slope is `1.0032895`; mean absolute relative error is 10%; and multiplicative RMSE is `exp(0.141128)=1.1516`.
- The `(H=256,mu=.9)` point is a real exception, not to be averaged away: its winning eta is `2.5x` the first-order prediction. It is one reason A5 retains a second-order buffer-geometry test.

The registered interpretation is therefore: the first-order deflation is strong and close to exact over most of the observed grid, especially all `mu=.5` rows, but it is not an identity of the empirical optimum. The paper may call it the dominant baseline explanation only while reporting the long-H/high-momentum exception.

## 3. House rules carried forward

### 3.1 Provenance and launch authority

Before any new scientific launch, freeze and hash the clean pushed Git commit, container/image, model revision, data, tokenizer, train/development/audit splits, exact cell registry, exact argv per cell, tuning grid, randomization plan, retry policy, evaluation commands, cost ledger, and analysis code. Placeholders are not launch authority.

All expected cells, including fixed-eta controls, tuning candidates, failures, divergences, and superseded retry attempts, remain in an append-only manifest. No historical absolute loss may replace a live within-seed control.

### 3.2 Spot-only execution and resource ceiling

- Spot/preemptible capacity only; on-demand fallback is forbidden.
- The campaign-wide ceiling remains 16 attached A100-equivalent accelerators.
- For A1/A3/A4, prefer reviewed Spot `a2-highgpu-1g` and survival-weighted zones, with `us-east1-b`, `us-west4`, and `us-west1` early in the rotation; midday `us-central1` is a hostile fallback. A6 paper-faithful stacks may use reviewed Spot H200/A100 equivalents but never on-demand.
- A1, A3, and A4 have `width_cap=2`: at most two independent atomic blocks may overlap. Capacity may degrade loss-blindly below that cap but may never expand above it.
- A provisioned accelerator may not remain without scientific work for more than 600 seconds.
- Every physical generation has numeric instance/disk identity, ownership nonce, lifecycle-final record, exact-ID teardown, and a zero-accelerator census. Name-, prefix-, wildcard-, or label-only deletion is forbidden.
- A block pauses before launch if its forecast would exceed its registered dollar ceiling by more than 25%. Cost is never a post-outcome exclusion rule.
- Each of A1, A3, and A4 has a separate `abort_burn_kill=$40.00`. Track cumulative pre-science aborted-launch spend independently from scientific stage spend; if it exceeds `$40.00` for a stage, stop and publish status rather than relaunching into launch bugs.
- Four distinct launch-machinery defects are already on record: mass teardown, missing fail-fast, census race, and budget margin. If a fifth distinct launch-machinery defect appears, stop the parallel path and publish a status recommending the proven serial P1 controller; do not patch the parallel executor on live VMs.

### 3.3 Pairing, randomization, retries, and divergences

The independent unit is a complete training-seed block. Within a seed, all compared arms share initialization, data order, worker allocation, work/tokens, evaluation data, hardware class, and time block. Arm-to-machine, block order, and within-block order are materialized pseudorandomly and hashed before launch.

Allowed retries are loss-blind and mechanical only: provider Spot preemption, VM/host/GPU failure, process exit before a scientific divergence is recorded, missing/checksum-invalid required artifact, or a pre-unblinding validator/provenance failure. A triggering failure reruns the entire atomic comparison block from the frozen initial state. Completed peers remain `COMPLETED` with original losses and artifacts; rerun peers use the explicit peer-retry reason. Partial optimizer/checkpoint/tape state is never resumed.

A scientifically divergent hyperparameter is retained, receives infinite loss for tuning and inference, and never triggers an infrastructure retry. Poor finite loss, preference after unblinding, or a desired sign is never a retry trigger.

### 3.4 Seed registry and blinding

The original unopened seed partitions remain authoritative for A1:

| Role | Shuffle seeds | Training-seed rule |
|---|---|---|
| A1 development completion | `359, 373` | decimal concatenation: `359359`, `373373` |
| A1 fresh confirmation | `383, 397, 409, 421, 433, 443, 457, 461` | decimal concatenation, e.g. `383383` |

New seed partitions, repository-searched for prior study use before this registration, are:

| Block | Development/tuning seeds | Initial confirmation seeds | Precision-only expansion seeds |
|---|---|---|---|
| A2 9B | `2003, 2011` | `2017, 2027, 2029` | `2039, 2053, 2063` |
| A3 frontier extension | reuses A1 development `359,373` | none; prospective law-extension cells | none |
| A4 M-axis | `2069, 2081` | `2083, 2087, 2089` | `2099, 2111, 2113` |
| A5 surgery | none | `2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179` | `2203, 2207` |
| A6(i) OMR | `2213` | `2221, 2237, 2239` | `2243, 2251, 2267` |
| A6(iii) SlowMo | `2269, 2273` | `2281, 2287, 2293, 2297, 2309` | none |

For every four-digit shuffle seed `s`, the training seed is `int(str(s)+str(s))`. A registry hash over all seed-role assignments is sealed before launch. A seed cannot move between development and confirmation after any outcome is opened.

Development losses are hidden until the complete tuning block validates and seals. Confirmation follows train-all-then-audit-all:

1. Train every registered confirmation cell without mounting or naming the audit evaluation artifact; completed training rows have `loss=null` and sealed checkpoint hashes.
2. Resolve loss-blind retries and seal the complete checkpoint registry.
3. Create one loss-blind audit authorization binding the full registry, commands, order, and timestamps.
4. Evaluate the complete hidden batch in committed order; expose no aggregate, per-example loss, rank, prediction, or log excerpt until the bundle validates and seals.
5. Record one shared unblind time. A mechanical audit failure can rerun only the complete hidden audit batch from the same checkpoint registry.

A6(ii) is exempt only because its outcomes are already sealed and public inside this repository. It is labeled historical and never pooled as a fresh seed.

### 3.5 Tuning and statistical analysis

- Each method gets the same number of initial eta candidates and the same maximum of one loss-blind boundary extension. If one member of a paired method block extends, the paired method receives its corresponding extension so tuning budgets remain equal.
- A selected eta must be interior on the pooled development mean, with a worse sampled point on both sides. If the one registered extension still leaves a boundary winner, the method is `UNBRACKETED`; no tuned comparison is claimed.
- The development choice is the lowest pooled point estimate. Neighbor losses and intervals are reported; interval-based reselection is forbidden.
- Confirmation uses only the frozen selected eta. No confirmation loss can alter tuning, method parameters, restart period, surgery branch, kernel estimator, or analysis.
- Primary uncertainty is the seed-level paired difference. Use a robust Student-t model with seed as top-level cluster; report the raw paired values, mean, median, SD, exact/Student-t interval, and the hierarchical-model interval.
- Holm-adjust two-sided 95% intervals within each experiment’s co-primary family. Equivalence is decided by the adjusted CI, not by a nonsignificant difference test.
- A precision-only expansion is allowed only when the predeclared initial seed block is complete and the adjusted CI half-width exceeds `epsilon`. All registered expansion seeds are then run; the observed sign or mean cannot trigger or cancel expansion.
- Evaluation-example bootstrap intervals are supplementary and must never be called training-seed uncertainty.

## 4. Registered experiment set

## A1. Multi-seed 135M momentum anchors

**Execution controls.** `width_cap=2` concurrent atomic blocks; separate cumulative pre-science `abort_burn_kill=$40.00`.

**Question.** Do the large fixed-eta full-parameter momentum penalties reproduce, and do both short- and long-H effects collapse after independent per-arm tuning?

**Stack.** Preserve the frozen SmolLM2-135M full-parameter protocol: Capybara; 5,000 training rows; locked 1,024-row development and disjoint 1,024-row audit sets; sequence 128; M=4; persistent inner AdamW LR `0.001`; RDA merge; no delta correction; bf16 wire/f32 syncer; strict quorum, true barrier, version-matched anchor; 655,360 total training tokens; fixed windows; Spot only.

**Anchors.** The short pair is `(H=16,mu=0)` versus `(H=16,mu=.9)`. The long pair is `(H=256,mu=0)` versus `(H=256,mu=.5)`. The common fixed-eta audit uses `eta=.0875`, the legacy boundary at which both momentum penalties were large.

Development seed 347 is already observed and remains development-only. Complete the original three-seed development set with 359 and 373 using these exact candidates:

| Anchor arm | Eta candidates |
|---|---|
| H16, `mu=0` | `0.015467960838455726, 0.021875, 0.030935921676911452` |
| H16, `mu=.9` | `0.0013671875, 0.002734375, 0.00546875` |
| H256, `mu=0` | `0.030935921676911452, 0.04375, 0.061871843353822904` |
| H256, `mu=.5` | `0.0109375, 0.021875, 0.030935921676911452` |

If a pooled three-seed winner is on a boundary, add exactly one paired outward geometric extension before confirmation. No second extension is allowed.

The eight fresh confirmation seeds run, in one blinded campaign, the four fixed-eta cells and the four independently tuned cells. Co-primary tuned contrasts are H16 `.9-0` and H256 `.5-0`.

**A1 gate.** The audit momentum row passes if both fixed penalties reproduce with adjusted lower endpoints above zero and both tuned-effect adjusted CIs lie wholly inside `[-.010,+.010]`. This is “tuned-collapse confirmed with seed-level CIs.” Failure is classified by the outcome table rather than hidden.

## A2. Qwen3.5-9B LoRA transfer

**Question.** Does the fixed-versus-tuned audit conclusion transfer to the 9B LoRA stack?

**Stack.** Qwen3.5-9B, LoRA rank 2/alpha 4, Capybara, M=4, four fragments, strict full quorum, inner AdamW LR `0.001`, fixed local windows, delta correction off, bf16 wire/f32 syncer, 700k tokens, H16=320 commits and H256=20 commits. Before launch, create and hash new 1,024-row development and disjoint 1,024-row audit sets; the historical 64-row endpoints are design evidence only.

**Pairs.** H16 compares `mu=0` with `mu=.9`; H256 compares `mu=0` with `mu=.5`. The fixed audit uses common `eta=.175`.

Equal-budget development grids on seeds 2003 and 2011 are:

| Arm | Eta candidates |
|---|---|
| H16/H256, `mu=0` | `0.175, 0.28, 0.448` |
| H16, `mu=.9` | `0.014, 0.028, 0.056` |
| H256, `mu=.5` | `0.0875, 0.175, 0.35` |

One paired factor-two boundary extension is permitted. Confirmation starts with three paired seeds and expands mechanically to six only if an adjusted CI half-width exceeds `.009`.

**A2 gate.** Report fixed replication and tuned classification at both H. Cross-stack support for the audit thesis requires the fixed effects to reproduce and both tuned CIs to lie within `[-.009,+.009]`; a surviving or reversed effect is equally reportable and scopes A1 rather than invalidating it.

## A3. Tuned memoryless loss-versus-communication frontier

**Execution controls.** `width_cap=2` concurrent atomic blocks; separate cumulative pre-science `abort_burn_kill=$40.00`.

**Question.** What is the independently tuned `mu=0` loss-versus-H frontier, and can a frozen finite-history kernel law predict its new endpoints?

Existing seed-347 minima are retained without reselection:

| H | Existing winning eta | Existing tuned NLL |
|---:|---:|---:|
| 16 | `0.021875` | `2.046027520801208` |
| 64 | `0.021875` | `2.1074817812764506` |
| 256 | `0.04375` | `2.1486300765736375` |

Add two complete development blocks at each new horizon, using seeds 359 and 373:

| H | Registered eta cells |
|---:|---|
| 8 | `0.0109375, 0.015467960838455726, 0.021875, 0.030935921676911452` |
| 512 | `0.021875, 0.030935921676911452, 0.04375, 0.061871843353822904, 0.0875` |

If the pooled winner is on a boundary, permit one extension only: H8 adds `0.00546875` or `0.04375`; H512 adds `0.015467960838455726` or `0.12374368670764582`. A remaining boundary winner is reported as unbracketed.

### Matched-eta kernel capture

Use `mu=0,eta=.021875` tapes at H16, H64, and H256 from the existing campaign and the registered H8/H512 cells. Replay must reconstruct every applied outer step and ordered pseudo-gradient used by the estimator. This common eta removes the old 9B rho-law confound in which H64 was measured at `.28` while H16/H256 were measured at `.175`.

If any existing full-parameter tape fails exact replay or lacks the required merged-delta sequence, a loss-blind matched-eta recapture of H16/H64/H256 at `mu=0,eta=.021875,seed=347` is authorized. Recapture is triggered only by the mechanical tape gate, never by rho or loss. No new outcome from those recaptures enters eta selection.

For fragment `f` with `K_f(H)` outer updates, estimate the energy-weighted full finite kernel `rho_k(H,f)` and define

```text
V_H = sum_f [ K_f + 2 sum_{k=1}^{K_f-1} (K_f-k) rho_k(H,f) ]
K_H = sum_f K_f
```

Before any H8/H512 endpoint loss is evaluated or exposed, fit on all existing H16/H64/H256 `mu=0` curve points the no-H-intercept finite-history model

```text
L_hat(H,eta) = L_init - a K_H eta + b V_H eta^2,
a > 0, b > 0.
```

Freeze the estimator, PSD/kernel regularization, coefficients, uncertainty procedure, and predictions. The model implies

```text
eta_hat_star(H) = a K_H / (2 b V_H)
F_hat(H)        = L_init - a^2 K_H^2 / (4 b V_H).
```

Only then may the H8/H512 evaluation bundle be unblinded.

**A3 gate.** The quantitative law-fit gate passes only if:

1. every matched-eta kernel passes replay and coverage;
2. both new selected etas are interior and satisfy `abs(log2(eta_star/eta_hat_star)) <= 0.5`;
3. both new tuned losses fall inside the frozen 95% prediction intervals;
4. extension-point frontier RMSE is at most `.010` NLL; and
5. the complete frontier has the predicted ordering, including `F(8) <= F(16)` and `F(512) >= F(256)`.

If the law gate fails, the five-point tuned frontier remains the result and all claims that rho quantitatively predicts its shape are dropped.

## A4. Worker-count axis

**Execution controls.** `width_cap=2` concurrent atomic blocks; separate cumulative pre-science `abort_burn_kill=$40.00`.

**Question.** Does fixed-eta inflation and tuned collapse persist when worker count changes, or is it specific to M=4 averaging?

Use the A1 135M stack at `M in {1,4}`, `H in {16,256}`, and the same short `.9` and long `.5` comparisons. Both M values are rerun on the same A4 seeds; historical or A1 losses are not imported as live controls. Fixed eta is `.0875`.

The equal-budget tuning grids at both M values are:

| Arm | Eta candidates |
|---|---|
| H16, `mu=0` | `.0109375, .021875, .04375` |
| H16, `mu=.9` | `.0013671875, .002734375, .00546875` |
| H256, `mu=0` | `.021875, .04375, .0875` |
| H256, `mu=.5` | `.0109375, .021875, .04375` |

Two development seeds select etas separately for every `(M,H,method)`. Three confirmation seeds expand to six only by the precision rule.

**A4 gate.** Report fixed and tuned effects at each M plus the seed-paired `M x tuning-status x method` interaction. Broad M-axis support requires fixed effects at both M values and tuned equivalence at both; a surviving M=1 effect scopes the artifact claim to distributed/multi-worker pseudo-gradients.

## A5. Closed-loop buffer surgery

**Question.** After first-order scale is matched, does curvature-weighted transverse buffer geometry cause a second-order loss term that tuning cannot remove?

Use H16 and H256, `mu=.9`, and the frozen tuned etas from development. For each of eight independent seed trajectories, take six fixed checkpoints per H: two early, two middle, and two late. From the same checkpoint and exact current pseudo-gradient `delta_t`, branch the buffer into:

1. factual;
2. zero/reset;
3. same-norm aligned;
4. same-norm anti-aligned;
5. orthogonal;
6. transverse-sign-flipped;
7. random-rotated transverse;
8. history-shuffled.

Panel 1 preserves the factual eta. Panel 2 rescales eta per branch to match the factual final applied-step norm and, where algebraically feasible, the current-gradient aligned component. These scaling values are computed without loss access. Measure high-precision one-step loss for every branch in both panels. Run the 12-commit closed-loop continuation for every branch in the scale-matched Panel 2; Panel 1 is the unmatched one-step diagnostic. On a frozen two-checkpoint-per-seed/H subset, compute held-out gradients and HVPs.

Before any branch loss is exposed, freeze the prediction

```text
Delta L_pred = -eta g^T d + 0.5 eta^2 d^T H d
```

and the norm-only, aligned-only, and first-order-only comparator models.

**A5 gate.** The second-order term survives tuning/matching only if the coefficient on `d^T H d` is positive with a seed-clustered 95% interval excluding zero, the full quadratic model lowers held-out seed-clustered RMSE by at least 20% versus the strongest comparator, and its predicted ordering is correct for at least six of the eight named interventions in both one-step and closed-loop panels. Otherwise retain the exact algebra but drop the causal second-order-loss claim.

The optional variance-only expansion from eight to ten trajectories uses seeds 2203 and 2207 and can trigger only from interval width, not effect direction.

## A6. Audit reproductions

Every A6 row uses the same reporting sentence:

> The published effect [replicates/does not replicate] at fixed eta and [survives/collapses/reverses/is inconclusive] under independently tuned per-arm learning rates with paired seeds.

### A6(i). Outer-Momentum Restarting bridge

Reproduce Outer-Momentum Restarting (`arXiv:2605.28585`) on the authors’ Llama-150M/C4 stack: sequence length 2,048; M=2; two replicas; about 3.3B tokens/12,500 inner steps; inner AdamW LR `1e-3`; H in `{64,2048}`; Nesterov outer momentum.

Use the paper’s exact normalized-buffer Nesterov recurrence rather than silently substituting the repository’s unnormalized PyTorch-style Nesterov:

```text
m_{t+1} = beta_out m_t + (1-beta_out) g_t
d_t     = (1+beta_out) m_{t+1} - beta_out m_t
theta   = theta - nu d_t
```

A hard restart sets only `m` to zero after every K completed outer rounds; model, inner optimizer, scheduler, and data state continue unchanged.

The fixed published transfer anchor is `nu=.9, beta_out=.7`, selected by the paper at H128. The hard-restart arm zeros only the outer buffer every `K=3` outer rounds, matching the paper’s published hard-restart recipe/figure; the no-restart arm is otherwise identical.

Per-arm tuning fixes `beta_out=.7` and `K=3` and gives both methods the exact eta grid

```text
nu in {.1,.3,.5,.7,.9,1.1}.
```

One loss-blind boundary extension (`.05` or `1.3`) is allowed for both methods together. Development seed 2213 selects eta independently by method and H. Three paired confirmation seeds expand to six only by the OMR epsilon precision rule.

**A6(i) gate.** At each H, first classify the fixed `.9/.7` restart effect, then classify the independently tuned difference. The paper’s robustness-over-grid claim is secondary and is reported as divergence rate and loss dispersion over the common eta grid; it cannot substitute for tuned minima.

### A6(ii). SCAFFOLD-lite sealed audit row

No new compute is authorized. Cite the sealed 9B record:

- fixed-eta SCAFFOLD-lite at H16: `1.47322` versus matched SGD-0.28 control `1.54547`, an apparent improvement of `0.07225`;
- exp2-52 frozen memoryless deconfound: tuned `eta=0.749500521952363` gives `1.4723387220473634` versus fresh `eta=.28` control `1.543579193244204`, an improvement of `0.0712404711968406`.

The tuned memoryless improvement is 98.60% of the magnitude of the earlier SCAFFOLD-lite fixed-baseline effect; the difference in improvement magnitudes is about `.00101`. Because these are sealed historical runs rather than a fresh multi-seed paired campaign, the row is labeled `HISTORICAL_MAGNITUDE_COLLAPSE`, not seed-level confirmation. Absolute endpoints are not cross-pooled beyond the sealed deconfound’s own pairing.

### A6(iii). SlowMo normalized-EMA-equivalent reproduction

Choose SlowMo (`arXiv:1910.00643`, ICLR 2020) because it is a canonical published outer-momentum method with public code, an exact short recurrence, a reported five-seed result, and an algebraic normalized-EMA reparameterization that makes the LR-confound audit especially sharp. It is also independent of the DiLoCo/Nesterov and restart families.

The exact SlowMo outer step is

```text
u_{t+1} = beta u_t + (x_{t,0}-x_{t,tau})/gamma_t
x_{t+1,0} = x_{t,0} - alpha gamma_t u_{t+1}.
```

With `m_t=(1-beta) gamma_t u_t` under a constant fast LR, this is a normalized EMA `m_{t+1}=beta m_t+(1-beta)delta_t` followed by effective outer LR `alpha/(1-beta)`. That reparameterization is why a separately tuned LR baseline is mandatory.

Reproduce the paper’s CIFAR-10 Local-SGD row: ResNet-18, 200 epochs, global batch 4,096, 32 logical workers, local Nesterov `.9`, weight decay `1e-4`, warmup 5 epochs, decays at 100/150/175, local period `tau=12`, SlowMo `alpha=1,beta=.7`, and reset local momentum buffers at the outer boundary. To retain the 16-accelerator house ceiling, run two deterministic logical workers per Spot A100 and require a loss-blind packing/parity canary against one-worker-per-device semantics before scientific launch.

The published fixed row is Local SGD `91.73%` versus SlowMo `93.20%`. Bind the exact authors-code argv and their `.025` reference fast-LR setting, including the paper’s worker-count scaling, before outcomes.

For the tuned audit, both methods receive the same reference fast-LR grid used by the paper:

```text
{.01,.025,.05,.1,.15}
```

SlowMo keeps `alpha=1,beta=.7`; no beta search is allowed. Two development seeds select LR independently. Five fresh paired seeds evaluate both the fixed published setting and each method’s selected LR.

**A6(iii) gate.** Fixed replication requires a paired mean SlowMo gain of at least `0.50` accuracy points with the 95% lower endpoint above zero. Tuned collapse requires the paired tuned-effect CI to lie in `[-0.25,+0.25]` points; survival requires the lower endpoint above zero and mean gain at least `0.25` points.

## 5. Full gate table

| Gate | Evidence unit | Pass rule | Failure/downgrade |
|---|---|---|---|
| G0 provenance | complete campaign | All hashes, expected cells, work, live controls, blind audit, replay, and teardown pass | Block scientific interpretation until repaired loss-blindly |
| G1 eta-law verification | six P1 momentum winners | Report slope/RMSE and all row residuals; no binary pass | First-order folklore is descriptive only; long-H `.9` exception remains explicit |
| G2 A1 fixed replication | 8 paired seeds, two anchors | Both fixed penalties have Holm-adjusted lower endpoints `>0` | `FIXED_NOT_REPLICATED`; audit still reportable |
| G3 A1 tuned collapse | 8 paired seeds, two anchors | Both adjusted tuned CIs wholly within `[-.010,+.010]` | Survives/reverses/inconclusive per frozen labels |
| G4 A2 transfer | 3–6 paired seeds | Fixed effects reproduce and tuned CIs lie in `[-.009,+.009]` at both H | Scope A1 to 135M or report surviving 9B effect |
| G5 A3 kernel integrity | exact replayed tapes | Complete matched-eta kernels at all five H; prediction frozen before endpoint unblind | Run only registered recapture; otherwise kill law fit |
| G6 A3 eta prediction | H8/H512 extension blocks | Both interior optima within `0.5` log2 units of prediction | Keep empirical frontier; drop eta-law claim |
| G7 A3 frontier prediction | H8/H512 endpoint bundle | Both inside 95% PIs, RMSE `<=.010`, correct endpoint ordering | Keep empirical frontier; drop quantitative rho-shape claim |
| G8 A4 M-axis | 3–6 paired seeds per M | Fixed effects reproduce and tuned equivalence holds at M1 and M4 | Scope artifact thesis to passing M values |
| G9 A5 second-order surgery | 8–10 seed trajectories | Positive curvature coefficient, `>=20%` held-out RMSE gain, `>=6/8` branch orderings at both horizons | Retain algebra; drop causal second-order claim |
| G10 A6(i) OMR fixed | 3–6 paired seeds per H | Fixed `.9/.7/K3` effect has published direction with CI excluding zero | Report nonreplication; do not tune away the failed replication |
| G11 A6(i) OMR tuned | same | Classify tuned difference with prebound compute-equivalent epsilon | Survive/collapse/reverse/inconclusive are all final outcomes |
| G12 A6(ii) SCAFFOLD-lite | sealed historical record | Cite fixed gain and 98.60% memoryless accounting | Historical/descriptive only; never call multi-seed confirmation |
| G13 A6(iii) SlowMo fixed | 5 paired seeds | Mean gain `>=.50` points and lower endpoint `>0` | Report nonreplication |
| G14 A6(iii) SlowMo tuned | 5 paired seeds | Tuned CI inside `[-.25,+.25]` points for collapse, or survival rule | Report frozen classification |
| G15 cost/Spot | ledger and provider evidence | Spot only, <=16 accelerators, within block ceiling plus registered reserve | Pause before new launch; never exclude completed results |

## 6. Kill and downgrade table

| Trigger | Required conclusion/action |
|---|---|
| A1 fixed effects fail | The legacy fixed-eta full-parameter effect is not seed-robust; the audit becomes a baseline-quality correction, not an artifact-collapse claim |
| A1 tuned effect survives | Momentum has a real tuned effect on the 135M anchors; retire the universal collapse thesis and report its magnitude/sign |
| A2 differs from A1 | Scope conclusions by stack/PEFT/full-parameter regime; no universal statement |
| A3 extension is unbracketed | Report only sampled frontier bounds; no extrapolated eta or law pass |
| A3 kernel law misses | The tuned frontier remains primary; rho is diagnostic, not a quantitative predictor of frontier shape |
| A4 shows an M interaction | State exactly which M values collapse/survive; do not average away the interaction |
| A5 quadratic model fails | First-order tuning explains the record; buffer geometry is not established as a causal loss term |
| OMR fixed effect fails | The published restart effect does not reproduce on the exact bridge; tuned comparison is still reported as a separate audit |
| OMR/SlowMo effect survives tuning | The audit finds a genuine method contribution; revise the thesis to “fixed-eta exaggerates effects” rather than “effects collapse” |
| SCAFFOLD source artifacts cannot substantiate the sealed numbers | Remove the audit row; do not spend new compute under this registration |
| Any confirmation outcome leaks before complete seal | Entire exposed seed block becomes development; register new untouched confirmation seeds before inference |
| Any method-specific tuning budget differs | Affected row is inadmissible until rerun with equal budgets |
| Cost ceiling is forecast to break | Stop before launch and amend prospectively; never terminate based on observed loss |

## 7. Cost registry

Costs are prospective planning ranges, not outcome-dependent caps. Revision 1.1 re-costed A1, A3, and A4 from the sealed cumulative P1 adaptive phase manifest, SHA-256 `8286d70fe11e47eed2e43f7fda4a9254370d9f20518ffa75aa22fc7b13789536`. The timing sample includes only completed, non-injected `a2-highgpu-1g` attempts lasting at least five minutes. It contains 22 valid H16 arms, 22 H64 arms, and 13 H256 arms. Their median wall times are:

| H | Median one-1g arm wall time |
|---:|---:|
| 16 | `34.10149415 min = 0.5683582358 h` |
| 64 | `14.543221175 min = 0.2423870196 h` |
| 256 | `9.7798688 min = 0.1629978133 h` |

H8 and H512 are not present as completed historical arms. The prospective extrapolation is frozen before either endpoint is run: least squares on the three medians with `minutes(H)=a+b/H` gives `a=8.1079407625`, `b=415.7045252571`, H8 `60.0710064196 min = 1.0011834403 h`, and H512 `8.9198636634 min = 0.1486643944 h`.

The cheapest registered survival-weighted option is Spot `a2-highgpu-1g` in `us-west4` at `$1.817/VM-h`. The revision 1.1 A3 and A4 lower bounds include every registered conditional science branch needed to prevent a later loss-blind boundary, recapture, or precision trigger from becoming mechanically unaffordable. Their hard ceilings remain the lower bound times `1.30`, rounded upward to the cent. The reviewed remaining-work guard retains its 25% factor; the final five percentage points cover realized boot, transient-provider, and exact-teardown burn.

Revision 1.1's A1 arithmetic was not a survivable complete-path bound. It horizon-weighted 48 H16 and 48 H256 arms and produced `$63.7859491841`, then capped A1 at `$82.93`. The operator's observed complete-path accounting places the cheapest one-GPU path at approximately `$106`, so revision 1.2 supersedes that A1 model and sets the hard ceiling directly to `$140.00`. This is approximately 32.1% above the observed path and supplies real Spot-churn/boot/teardown margin. The prior formula is retained below as superseded history rather than silently rewritten.

The arithmetic is:

```text
A1 revision-1.1 modeled hours = 48*0.5683582358 + 48*0.1629978133
                              = 35.1050903600 h
A1 revision-1.1 modeled lower = 35.1050903600*$1.817 = $63.7859491841
A1 revision-1.1 cap           = $63.7859491841*1.30 = $82.9217339394 -> $82.93
A1 observed cheapest complete 1g path ~= $106
A1 revision-1.2 hard ceiling          = $140.00

A3 hours = 10*1.0011834403 + 12*0.1486643944
         + (0.5683582358+0.25)
         + (0.2423870196+0.12)
         + (0.1629978133+0.06)
         = 13.1995502047 h
A3 lower = 13.1995502047*$1.817 = $23.9835827219
A3 cap   = $23.9835827219*1.30 = $31.1786575385 -> $31.18

A4 hours = 80*0.5683582358 + 80*0.1629978133
         = 58.5084839333 h
A4 lower = 58.5084839333*$1.817 = $106.3099153069
A4 cap   = $106.3099153069*1.30 = $138.2028898989 -> $138.21
```

The 9B estimate still uses the sealed empirical record of about `$8` for six LoRA arms. A6 paper-faithful rows still require a pre-launch timing canary, so their dollar ranges remain wider. The A6(i) OMR planning range remains `$250–800`; only its reserve above that range is reduced by the program-ceiling reallocation.

| Block | Planned new work | Planned accelerator budget | Spot-dollar planning range | Hard block ceiling |
|---|---:|---:|---:|---:|
| A1 | 24 development + up to 8 paired boundary-extension + 64 confirmation = max 96 arms | observed complete-path correction | `~$106.00–132.50` | `$140.00` |
| A2 | 24 tuning + 24 initial confirmation = 48 9B LoRA arms; max 72 with precision expansion | empirical arm-cost basis | `$65–125` | `$160` |
| A3 | 18 frontier arms; max 22 with boundary extension and 25 with mechanical recaptures | `13.1995502047` A100-h lower bound including capture overhead | `$23.9835827219–29.9794784024` | `$31.18` |
| A4 | 48 development + up to 16 boundary-extension + 48 initial confirmation + 48 precision expansion = max 160 arms | `58.5084839333` A100-h lower bound | `$106.3099153069–132.8873941336` | `$138.21` |
| A5 | 16 parent trajectories, 1,536 one-step branch evaluations across two scale panels, 768 scale-matched twelve-commit continuations, frozen HVP subset | `90–180` A100-h | `$100–250` | `$320` |
| A6(i) OMR | 24 tuning + 24 initial confirmation = 48 two-replica runs; max 72 | `120–360` A100-equivalent h after canary | `$250–800` | `$945.61` |
| A6(ii) SCAFFOLD-lite | sealed citation only | `0` | `$0` | `$0` |
| A6(iii) SlowMo | 20 tuning + 20 confirmation = 40 16-A100 packed distributed runs | `160–480` A100-h after canary | `$180–600` | `$750` |
| **Program total** | excludes already sealed evidence and the operator-excluded `$9.231503` A1 infrastructure burn | corrected A1 planning range plus unchanged stage planning ranges | **`$831.2934980288–2,070.3668725360`** | **`$2,485`** |

The revision 1.2 sum of individual block hard ceilings is exactly `$2,485.00`: A1 `$140.00` + A2 `$160.00` + A3 `$31.18` + A4 `$138.21` + A5 `$320.00` + A6(i) OMR `$945.61` + A6(ii) SCAFFOLD-lite `$0.00` + A6(iii) SlowMo `$750.00`. This equals, and therefore fits, the unchanged `$2,485.00` program hard ceiling. No program headroom remains unallocated.

The `$54.39` A6(i)-to-A1 reserve transfer is authorized only by revision 1.2. No further stage may borrow another stage’s unused ceiling without a prospective amendment. A6(i) and A6(iii) can be independently killed for reproducibility or cost before scientific outcomes; neither is required to interpret A1–A5.

## 8. Reporting order and main-paper objects

1. Report the P1 eta-scaling fit and the exact outlier.
2. Report fixed and tuned effects side by side for every audit row; never show only the tuned winner or only the fixed effect.
3. Make the five-point tuned `mu=0` loss-versus-communication frontier the primary performance figure, with communication count/bytes and total work shown explicitly.
4. Overlay the frozen A3 kernel-law prediction and its prediction intervals; if it fails, show the miss.
5. Report seed-level paired points and CIs before pooled/hierarchical summaries.
6. Report A5 as the residual second-order test after first-order matching, not as a rescue of the retired poison headline.
7. Put A6 in an audit table with columns: published method/effect, exact recurrence, fixed-eta replication, tuning budget, tuned effect, classification, seeds, compute, and deviations from the source paper.
8. Publish the full cost ledger, all failed/diverged arms, retry lineage, and Spot teardown evidence.

The strongest permitted conclusion if the central gates pass is:

> Across full-parameter and PEFT DiLoCo-style anchors, fixed-eta comparisons substantially overstate outer-optimizer effects. Independent per-arm LR tuning with paired seeds collapses the momentum contrasts, and the relevant systems object becomes the tuned loss-versus-communication frontier. A finite-history kernel model predicts that frontier prospectively, while controlled buffer surgery identifies the smaller curvature-weighted term that remains after first-order tuning.

If any clause fails, delete only that clause and report the registered alternative classification. No controller, optimizer zoo, or post-hoc method search is authorized by this program.
