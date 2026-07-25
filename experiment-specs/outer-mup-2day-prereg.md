# Outer-muP two-day preregistration

**Program ID:** `outer-mup-2day`

**Status:** `PREREGISTERED`, prospective for all scientific outcomes described below.

**Registered:** 2026-07-24.

**Machine-readable companion:** `experiment-specs/outer-mup-2day-prereg.json`.

This is a two-day, gate-driven test of whether measured pseudo-gradient persistence replaces the usual worst-case outer-momentum learning-rate multiplier, and whether that replacement transfers tuned outer learning rates across sync horizon, worker count, and model scale. E0 is the code prerequisite; E1 is the only discriminator gate that can authorize the transfer program. If E1 fails, the program stops and becomes a tuned-baseline audit paper. No later result may rescue G1.

The required sequence is:

```text
E0 instrumentation
  -> E1 135M discriminator
  -> G1
       FAIL / NOT_EVALUABLE -> STOP; audit paper
       PASS -> E2 M-axis || E3 loss-blind probes
            -> hash-locked E3 prediction commit
            -> E3 verification
            -> G2
                 FAIL / NOT_EVALUABLE -> STOP; audit paper
                 PASS -> E4 boundary || E5 real-interconnect demo
                      -> optional E6 SNOO deflation and E7 Lean lemma
```

## 1. Theory position and discriminating estimand

> Khaled et al. 2509.10439 Thm 2's 1/(1-mu) multiplier is the rho->1 limit of the registered filter law eta_eff=eta(1+mu/(1-mu*rho)); E1 discriminates them.

For momentum `mu` and lag-1 pseudo-gradient persistence `rho`, define

```text
A(mu,rho) = 1 + mu / (1 - mu*rho)
eta_eff   = eta * A(mu,rho).
```

At `rho=1`, `A(mu,1)=1/(1-mu)`, so matching a memoryless tuned learning rate gives the familiar first-order prediction

```text
eta_star(mu,H) / eta_star(0,H) = 1 - mu.
```

The registered finite-persistence prediction is instead

```text
eta_star(mu,H) / eta_star(0,H) = 1 / A(mu,rho_hat(H)).
```

At `mu=.9`, E1 compares the observed deviation factor

```text
D_obs(H) = [eta_star(.9,H) / eta_star(0,H)] / .1
```

with

```text
D_pred(H) = {1 / A(.9,rho_hat(H))} / .1.
```

`D=1` is the `rho=1` first-order prediction. The high-H confidence interval and four-H rank pattern below are the discriminator; the theory sentence is not treated as true merely because the algebraic limit is true.

## 2. Authority, immutability, and closed outcomes

Before a scientific process starts, the launch manifest must bind a clean pushed Git commit; raw SHA-256 hashes of this Markdown and its JSON companion; exact model revisions and model-file hashes; data, tokenizer, train/development split, row, packed-token, and example-identity hashes; a complete expected-cell registry; the exact argv hash of every cell; the randomization order; and the retry policy. A placeholder, an unpushed commit, an incomplete cell registry, or an unregistered command is not launch authority.

Outcome-aware amendments are forbidden. A prospective amendment must precede all affected outcomes, be committed, retain the superseded bytes, and be limited to a non-scientific typo, a loss-blind hardware substitution with the same numerical contract, or a clarification that only strengthens a fail-closed evidence check.

The closed cell statuses are:

| Status | Meaning |
|---|---|
| `PLANNED` | Registered, not started |
| `RUNNING` | Scientific process started |
| `COMPLETED` | Full work, finite loss, and all required evidence validate |
| `SCIENTIFIC_DIVERGENCE` | Registered run diverged; retained and assigned infinite tuning loss |
| `INFRA_FAILURE` | Loss-blind infrastructure failure eligible for the registered whole-curve retry |
| `INVALID_WORK` | Command, work, telemetry, loss, or artifact contract failed |
| `NOT_RUN_GATE_STOP` | A prior gate stopped the program |
| `NOT_RUN_WALL_CEILING` | The registered wall-clock ceiling stopped remaining work |
| `NOT_RUN_OPTIONAL_NO_SLACK` | Optional stage lacked preregistered slack |

Gate verdicts are only `PASS`, `FAIL`, or `NOT_EVALUABLE`. Program verdicts are only `CORE_COMPLETE`, `CORE_COMPLETE_WITH_OPTIONALS`, `STOP_G1_FAIL_AUDIT_PAPER`, `STOP_G1_NOT_EVALUABLE_AUDIT_PAPER`, `STOP_G2_FAIL_AUDIT_PAPER`, `STOP_G2_NOT_EVALUABLE_AUDIT_PAPER`, or `STOP_WALL_CEILING_AUDIT_PAPER`.

## 3. Common numerical and work contract

Unless a stage explicitly overrides a field, every scientific verification arm uses:

| Field | Registered value |
|---|---|
| Model family | `HuggingFaceTB/SmolLM2-{135M,360M,1.7B}`; exact revision/hash bound before stage launch |
| Tuning | Full parameter |
| Data | Pinned local `trl-lib/Capybara`; 5,000 train rows and 1,024 disjoint development rows |
| Outcome | Development NLL per token, finite and recorded at full precision |
| Sequence / microbatch | 128 / 1 |
| Fragments | 4 |
| Inner optimizer | Persistent AdamW, LR `.001` |
| Outer optimizer | Repository Nesterov |
| Merge | RDA; delta correction off |
| Arithmetic | bf16 wire; f32 syncer |
| Synchronization | Strict full quorum, true barrier, version-matched anchor, fixed windows, zero injected delay/jitter |
| Verification work | 1,310,720 total training tokens per arm |
| Telemetry | `yeto_rho_telemetry_v1`, one row per committed outer round |

For a listed scientific seed `s`, the shuffle seed is `s` and the training seed is `int(str(s)+str(s))`. A complete seed curve—not an evaluation example—is the independent unit.

### 3.1 Work evidence

A cell is `COMPLETED` only if all of the following exist and validate:

- the exact registered command hash and clean pushed Git commit;
- every registered learner inner step and every registered global outer commit;
- a finite evaluation loss and hashed, row-aligned per-example evaluation output;
- a checkpoint and checkpoint SHA-256;
- `rho-telemetry.jsonl`, schema `yeto_rho_telemetry_v1`, exactly one record per outer commit, and its SHA-256;
- immutable attempt ID and start/end timestamps.

Partial work never counts. A divergence remains in the registry with infinite tuning loss but does not satisfy the finite-loss work requirement. Missing, invalid, divergent, and wall-stopped cells are never silently removed from a registered gate denominator.

Retries are allowed only for a loss-blind lost/preempted host, hardware failure, process exit before scientific divergence, checksum-invalid required artifact, or loss-blind validator failure. The retry reruns the entire paired seed curve from the frozen initial state. Poor finite loss, divergence, a desired sign, or post-unblind preference can never trigger a retry. Partial optimizer, checkpoint, or telemetry state is never resumed.

## 4. Eta curves, selection, and confidence intervals

Every registered four-point eta ladder is log-symmetric around a frozen center `c`:

```text
log2 offsets = [-.75, -.25, +.25, +.75]
eta_j         = c * 2^(offset_j).
```

The multipliers are exactly

```text
[0.5946035575013605,
 0.8408964152537145,
 1.189207115002721,
 1.681792830507429].
```

No one-sided or outcome-aware extension is allowed.

For each pooled curve, let `x=log2(eta)` and fit `loss=a*x^2+b*x+c0` to the four seed-mean losses. An eta optimum is bracketed only when `a>0` and the vertex `-b/(2a)` is strictly inside the numeric grid. Then `eta_star=2^(-b/(2a))`. Nonpositive curvature or a vertex on/outside a boundary is `UNBRACKETED`. A missing, invalid, or nonfinite required seed cell invalidates the curve instead of being dropped. The lower eta wins any discrete diagnostic tie.

All registered 95% seed confidence intervals use 10,000 paired nonparametric resamples of complete seed curves with RNG seed `20260724`, refitting `eta_star` on every resample. Intervals are computed on `log2 eta` or the paired log ratio. Evaluation-example bootstraps may be shown only as secondary diagnostics.

The primary `rho_hat(H)` is taken from `mu=0` telemetry at the eta grid point nearest the pooled `eta_star(mu=0)`: average within fragment, average the four fragment estimates, then average the five primary seed estimates. Lag 1 is the gate input. Lags 2–4 must be reported as kernel/AR(1)-adequacy diagnostics and cannot replace lag 1.

## 5. E0 — instrumentation and probe prerequisite

E0 is implemented in two pushed commits:

- `a49b528`: opt-in syncer telemetry, direct and harness flags, exact norms/cross-worker cosines, and four per-fragment projected lag histories;
- `3f6c5c1`: `scripts/rho_probe.py`, short-run command contract, fail-closed JSONL validation, bootstrap report, and M=8 harness preset.

Telemetry is off unless `--rho-telemetry` is supplied. It uses a deterministic 4,096-dimensional CountSketch per tensor group with seed `0x5945544f52484f31`, retaining only four sketches per fragment. It records lag-1 through lag-4 projected cosine estimates, exact merged and worker pseudo-gradient L2 norms, and exact pairwise cross-worker pseudo-gradient cosines. Its schema is `yeto_rho_telemetry_v1`. The short probe emits `yeto_rho_probe_report_v1`.

E0 exits only when the telemetry/probe tests are green and both commits are ancestors of the experiment branch.

## 6. E1 — 135M discriminator grid

E1 uses SmolLM2-135M, `M=4`, `H in {16,64,256,512}`, and full eta curves for `mu in {0,.9}`. It additionally runs `mu=.5` at H16 and H256.

The five primary paired seeds are `{101,103,107,109,113}`. The three prespecified contested-cell top-ups `{127,131,137}` run every eta at `(H256,mu=.9)` and `(H512,mu=.9)`. Top-ups are a secondary robustness report only: they cannot enter the five-seed G1 confidence interval, the four-H Spearman calculation, or change G1.

### 6.1 Frozen centers and four-point ladders

| H | mu | Center | Four registered eta values |
|---:|---:|---:|---|
| 16 | 0 | `.021875` | `.01300695282034226, .018394609083675004, .02601390564068452, .03678921816735001` |
| 16 | .5 | `.0109375` | `.00650347641017113, .009197304541837502, .01300695282034226, .018394609083675004` |
| 16 | .9 | `.002734375` | `.0016258691025427825, .0022993261354593755, .003251738205085565, .004598652270918751` |
| 64 | 0 | `.021875` | `.01300695282034226, .018394609083675004, .02601390564068452, .03678921816735001` |
| 64 | .9 | `.002734375` | `.0016258691025427825, .0022993261354593755, .003251738205085565, .004598652270918751` |
| 256 | 0 | `.04375` | `.02601390564068452, .03678921816735001, .05202781128136904, .07357843633470001` |
| 256 | .5 | `.021875` | `.01300695282034226, .018394609083675004, .02601390564068452, .03678921816735001` |
| 256 | .9 | `.0109375` | `.00650347641017113, .009197304541837502, .01300695282034226, .018394609083675004` |
| 512 | 0 | `.04375` | `.02601390564068452, .03678921816735001, .05202781128136904, .07357843633470001` |
| 512 | .9 | `.0109375` | `.00650347641017113, .009197304541837502, .01300695282034226, .018394609083675004` |

H16/H64/H256 centers are sealed prior-campaign pooled winners. No completed H512 optimum exists; H512 prospectively inherits the nearest completed H256 winner. This fact must remain explicit and cannot be rewritten as H512 evidence.

### 6.2 Work and size

At M4 the 1,310,720-token budget gives each learner 2,560 inner steps:

| H | Global outer commits | Commits per fragment |
|---:|---:|---:|
| 16 | 640 | 160 |
| 64 | 160 | 40 |
| 256 | 40 | 10 |
| 512 | 20 | 5 |

There are ten eta-curve families, 200 primary runs, and 24 contested top-up runs: 224 E1 training runs in total. Every run must carry telemetry.

### 6.3 G1 — hard discriminator and stop rule

Using only the five primary seeds, G1 is `PASS` if and only if all of these hold:

1. At H256, the paired-seed 95% CI for `D_obs(H)` excludes `1.0`.
2. At H512, the paired-seed 95% CI for `D_obs(H)` excludes `1.0`.
3. Across H16, H64, H256, and H512, ordinary Spearman correlation with midranks between `D_pred(H)` and `D_obs(H)` is at least `.8`.
4. Every E1 primary work record is valid, every required eta optimum is interior, and lag-1 `rho_hat` is defined at all four H values.

A scientifically evaluable false condition is `FAIL`. Missing/invalid work, an unbracketed required optimum, or undefined rho is `NOT_EVALUABLE`.

`PASS` receives scientific verdict `RHO_FILTER_DISCRIMINATED`. Other descriptive E1 vocabularies are `RHO1_FIRST_ORDER_NOT_REJECTED`, `HIGH_H_DEVIATION_WITHOUT_RHO_ORDERING`, `RHO_ORDERING_WITHOUT_HIGH_H_DEVIATION`, and `NOT_EVALUABLE`; none authorizes continuation.

**G1 FAIL means the experimental program stops.** No E2–E7 scientific cell may launch. The fallback is a tuned outer-optimizer audit paper reporting all E1 curves, failures, telemetry, and the rejected discriminator. `NOT_EVALUABLE` also stops, producing an incomplete-evidence audit paper. The eight-seed contested summaries are secondary and cannot reverse this decision.

## 7. E2 — 135M M-axis

E2 runs only after G1 `PASS`. New worker counts are `M in {1,2,8}`; E1 supplies the M4 anchor. At H16 and H256, each new M crosses `mu in {0,.9}`, uses the same five primary seeds, and receives an independent four-point eta ladder centered on the sealed E1 M4 `eta_star` at that H/mu. This is 240 new runs.

The total token budget remains 1,310,720 per arm, giving learner-step counts `{M1:10240, M2:5120, M8:1280}`. All are exact full windows at H16/H256. The resulting 135M surface covers H `{16,256}`, M `{1,2,4,8}`, and mu `{0,.9}`.

For later target cells, the M4 `log2 eta_star` surface is piecewise linear in `log2 H` across all four E1 H values. At each M, the log2 ratio to M4 is observed at H16/H256; it is linearly interpolated to H64 and linearly extrapolated with that already-fixed slope to H512. No target-scale loss can alter this interpolation.

## 8. E3 — loss-blind probes, sealed transfer predictions, and G2

After G1, E2 and the loss-blind portion of E3 may overlap. E3 runs 24-global-round telemetry probes at each combination of:

```text
scale in {135M,360M,1.7B}
H     in {16,64,256,512}
M     in {1,4,8}
mu    in {0,.9}
seed  = 101
```

There are 72 probe cells. `scripts/rho_probe.py` must run them without endpoint loss evaluation and emit `yeto_rho_probe_report_v1`. The 135M probe eta is the E1/E2 interpolated optimum. The target-scale probe uses the same numeric 135M eta only to elicit loss-blind telemetry; it is not a target-scale tuning result.

### 8.1 Frozen outer-muP prediction rule

For each H/M coordinate, let `eta0_135(H,M)` be the preregistered E1/E2 mu0 surface. Let `N_S(H,M)` be the fragment-balanced geometric mean merged pseudo-gradient L2 norm from the scale-S, mu0 probe. Define

```text
eta0_pred(S,H,M) = eta0_135(H,M) * N_135(H,M) / N_S(H,M)
rho_S(H,M)       = fragment-balanced lag-1 rho from the scale-S mu0 probe
eta_pred(S,H,M,mu)
                 = eta0_pred(S,H,M) / A(mu,rho_S(H,M)).
```

For mu0, `A=1`. Cross-worker cosine and lags 2–4 are sealed diagnostics, not fitted degrees of freedom. No 360M/1.7B outcome may fit a coefficient, select a feature, or alter this equation.

### 8.2 Prediction hash lock

The prediction artifact is `experiment-specs/outer-mup-sealed-predictions.json`, schema `yeto_outer_mup_sealed_predictions_v1`. It must enumerate exactly 48 cells:

```text
scale in {360M,1.7B}
H     in {16,64,256,512}
M     in {1,4,8}
mu    in {0,.9}.
```

It must contain both contract hashes, its source commit, every probe-report path/hash, the E1/E2 selection hash, the exact prediction rule, all 48 cell IDs and predicted etas, a UTC creation time, and `verification_loss_seen=false`. Its canonical SHA-256 is recorded. The file must then be committed and pushed to the experiment branch.

Before any verification process launches, its authorization must cite the prediction Git commit and file SHA-256 and prove that the commit is reachable from the pushed experiment branch. The earliest verification process start must be later than the prediction commit. Any 360M or 1.7B endpoint loss exposed before this seal invalidates E3; deleting a log later cannot repair the ordering violation.

### 8.3 Verification and G2

Each of the 48 sealed cells receives the registered four-point ladder centered on its predicted eta and the five seeds `{101,103,107,109,113}`: 960 verification runs. Work is 1,310,720 total tokens per arm. The H512/M8 cell uses a registered 256-step terminal partial window after 1,024 full learner steps; every other target coordinate closes exact full windows.

A cell is `HIT_WITHIN_SEED_CI` exactly when its predicted `log2 eta` lies within the paired-seed bootstrap 95% CI of an interior empirical `eta_star`. An unbracketed curve is `MISS_UNBRACKETED`; invalid/missing work is `MISS_INVALID_OR_MISSING_WORK`; a valid interval miss is `MISS_OUTSIDE_SEED_CI`. All 48 cells stay in the denominator.

G2 is `PASS` if and only if the prediction ordering contract holds, no target-scale loss preceded the seal, E3 verification finishes within 14 hours, and at least 75% of the sealed cells hit. With 48 cells, the threshold is exactly 36 hits. Fewer than 36 hits in a complete registry is `FAIL`; an ordering violation or incomplete/invalid registry is `NOT_EVALUABLE`. Either stops the program before E4/E5 and yields the audit paper with failed or incomplete transfer.

## 9. E4 — mu=.95 boundary and causal buffer pairs

E4 runs only after G2 `PASS`. At 135M/M4 it crosses `H in {256,512}` with `mu in {.9,.95}` and the five primary seeds. The mu=.9 E1 curves are reused. Each new mu=.95 four-point curve is centered at

```text
eta_star(mu=0,H) / A(.95,rho_hat(H)),
```

using only the sealed E1 estimates. This adds 40 boundary runs.

At the pooled selected eta for every H/mu, take three fixed checkpoints per seed (early, middle, late). From the same checkpoint and exact current pseudo-gradient, evaluate two causal branch pairs:

1. factual buffer versus zero buffer;
2. factual buffer versus a same-norm buffer aligned with the current pseudo-gradient.

Both branches receive a high-precision one-outer-step evaluation and an eight-commit closed-loop continuation. There are 120 registered checkpoint/pair units. Full finite loss and telemetry are required for both branches; a pair with one invalid branch is invalid, not one-sided evidence.

The closed E4 verdicts are `STABLE_FILTER_EXTENSION`, `MU095_BOUNDARY_BREAK`, `BUFFER_CAUSAL_EFFECT_ONLY`, `NO_BUFFER_CAUSAL_EFFECT`, and `NOT_EVALUABLE`.

## 10. E5 — two-node, real-interconnect 1.7B demo

E5 runs only after G2 `PASS` and may overlap E4. It is one 1.7B, H256, M8, mu=.9, seed-101 arm at the hash-locked E3 predicted eta. Exactly two physical nodes host four learners each. The registered work is 1,280 learner steps and 20 global commits at 1,310,720 total tokens.

This is not satisfied by a one-node process layout or loopback emulation. Evidence must include an immutable learner-to-node map; non-loopback interface identities on both nodes; cross-node flow endpoints and byte counters spanning the run; proof that at least one admitted worker pseudo-gradient crossed nodes at every sync; eight-worker rho telemetry; finite evaluation loss; and exact full step/commit counts.

## 11. E6 — slack-contingent 135M SNOO deflation

E6 is optional and cannot change G1, G2, or the core verdict. It launches only after every required stage is sealed, at least eight hours remain before the 48-hour deadline, and a loss-blind P90 forecast is at most six hours. Otherwise its mandatory status is `NOT_RUN_OPTIONAL_NO_SLACK` and verdict `SKIPPED_NO_SLACK`.

If authorized, E6 uses a pinned C4 pretraining materialization, SmolLM2-135M full parameter, sequence 2,048, M1, `H in {100,400}`, `mu in {0,.9}`, the five primary seeds, and equal four-point eta tuning budgets: 80 tuning runs. The source-paper 135M/M1/H-specific fixed mu=.9 recipe must also be transcribed and hash-bound before launch and evaluated without replacing either method's equal tuning grid. Its estimand is independently tuned `mu=.9` minus independently tuned `mu=0` endpoint loss at each H, with an equivalence margin of `.01` loss. It is labeled a SNOO-style deflation audit unless every source-paper configuration field is separately bound; it is not silently promoted to an exact replication.

Closed E6 verdicts are `SNOO_GAIN_SURVIVES_TUNING`, `SNOO_GAIN_DEFLATED`, `SNOO_FIXED_NOT_REPLICATED`, `SNOO_REVERSED`, `NOT_EVALUABLE`, and `SKIPPED_NO_SLACK`.

## 12. E7 — optional Lean lemma

E7 is optional, outcome-independent, and may run in parallel. In the repository Lean project it attempts the statement:

```text
for real mu with 0 <= mu < 1,
Tendsto (fun rho => 1 + mu/(1-mu*rho)) (nhds 1) (nhds (1/(1-mu))).
```

The algebraic corollary is `1 + mu/(1-mu) = 1/(1-mu)`. The closed verdict is `PROVED`, `COUNTEREXAMPLE`, or `NOT_ATTEMPTED`. A proof must compile in the pinned Lean project, but E7 has no effect on any empirical gate.

## 13. Wall-clock ceilings

Nodes are free for planning purposes; elapsed wall time is not. The 48-hour program clock starts at the first E1 scientific process and stops at the E5 evidence seal or an earlier registered stop.

| Stage | Hard wall ceiling |
|---|---:|
| E1 | 14 h |
| E2 | 8 h |
| E3 loss-blind probes | 6 h |
| E3 prediction generation, validation, commit, and push | 1 h |
| E3 verification | 14 h |
| E4 | 8 h |
| E5 | 6 h |
| E6, if authorized | 6 h |

The ceiling is measured from the earliest stage process start through the final immutable stage evidence seal, including retries and validation. After G1, E2 and loss-blind E3 probes overlap. After G2, E4 and E5 overlap. E7 may overlap anything.

At a ceiling, terminate remaining stage work, retain every attempt, mark missing cells `NOT_RUN_WALL_CEILING`, and apply the registered `NOT_EVALUABLE`/stop rule. No extension is allowed after seeing outcomes. The specifically requested ceilings are hard: E1 must be at most 14 hours, and E3 verification must be at most 14 hours.

`CORE_COMPLETE` requires G1 `PASS`, G2 `PASS`, and valid complete E4/E5 evidence. `CORE_COMPLETE_WITH_OPTIONALS` requires those same core conditions plus completed E6 and/or E7 evidence. Optional work never substitutes for E4/E5. A required post-gate stage that cannot seal before its stage or program ceiling receives `STOP_WALL_CEILING_AUDIT_PAPER`.

## 14. Fallback paper and complete reporting

G1 failure, G1 non-evaluability, G2 failure, G2 non-evaluability, or a required wall-ceiling breach produces the tuned outer-optimizer audit paper. It must report every registered cell/status, finite loss, divergence, invalid attempt, eta bracket, unbracketed optimum, rho/norm/cross-worker telemetry, gate calculation, prediction-order artifact if E3 began, and wall/work-evidence failure. It may not claim rho-filter discrimination, universal eta prediction, or outer-muP transfer after G1 failure.

Negative, null, reversed, unbracketed, boundary, and failed-transfer outcomes are publishable. A later optional result cannot retroactively change a gate, remove a denominator cell, authorize a retry, or convert an audit verdict into a successful core program.
