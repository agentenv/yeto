# Best-paper phase map: frozen P0/P1 preregistration

**Frozen:** 2026-07-14, before any P1 scientific loss is opened

**Scope:** full-parameter SmolLM2 low-LR gate

**Machine-readable companion:**
`experiment-specs/best-paper-phase-map-p0-p1-prereg.json`

This protocol replaces EXP2.30 for all claims about tuned outer momentum on a
full-parameter model. Legacy EXP2.30 is design evidence only; its endpoints
must not be pooled into any estimate below.

## 1. Question and estimands

The first question is deliberately fatal to the current paper framing:

> After independently bracketing outer LR for each `(H, mu)`, does a genuine
> helpful-to-harmful outer-momentum transition remain?

For horizon `H` and momentum `mu`, define the development tuned endpoint

```text
T(H, mu) = minimum locked-eval NLL over the preregistered eta bracket.
```

The two primary development contrasts are:

```text
D_short = T(16, .9) - T(16, 0)
D_long  = min[T(256, .5), T(256, .9)] - T(256, 0)
```

Positive `D_short` means high momentum hurts at short H. Negative `D_long`
means some tested momentum helps at long H after LR tuning. H64 is the
predeclared intermediate-horizon check, not a substitute for either primary
corner.

P1 is development, not confirmation. A one-seed sign pattern can advance the
campaign but cannot appear in the paper as seed-robust evidence.

## 2. Frozen common protocol

The following must be identical across scientific arms except for the
registered `H`, `mu`, `eta`, seed, output path, and randomized order:

| item | frozen value |
|---|---|
| model | `HuggingFaceTB/SmolLM2-135M`, full parameter |
| model revision | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| data | local pinned `trl-lib/Capybara` parquet; hash bound before P0 |
| train/eval | 5,000 train rows; 1,024 common disjoint locked eval rows |
| sequence / microbatch | 128 / 1 |
| learners / fragments | M=4 / 4, full quorum |
| inner optimizer | AdamW, LR `0.001`, persistent state |
| outer optimizer | repository Nesterov implementation |
| merge | explicit `--matrix-merge rda`; resolved per-fragment modes sealed |
| correction | `--delta-correction none` |
| arithmetic | bf16 wire / f32 syncer |
| schedule | fixed padded windows, zero injected delay/jitter |
| synchronization | `--strict-quorum --barrier-sync --version-matched-anchor` |
| total work | 655,360 training tokens per arm |
| device | one GCP `a2-highgpu-4g` scientific VM, four A100 GPUs |
| provisioning | Spot only; no on-demand fallback |

For H16/H64/H256, respectively:

| H | fixed-window microsteps | fixed-window tokens | global commits |
|---:|---:|---:|---:|
| 16 | 16 | 2,048 | 320 |
| 64 | 64 | 8,192 | 80 |
| 256 | 256 | 32,768 | 20 |

The H256 result has only five buffer updates per fragment. P1 can answer the
fixed-budget LR gate, but it cannot validate a stationary autocorrelation law.
A later equal-outer-commit experiment is mandatory before making stationary or
universal-kernel claims.

Before P0, bind and seal a clean pushed 40-hex Git commit, numeric image ID
`7290368630472593484`, model-file hash, parquet hash, train-row hash,
evaluation-row hash, packed evaluation sequence/token hash, full command hash,
randomization-plan hash, and an independent retry-policy hash. A retry
authorization must cite the retry-policy hash; it must never reuse the
randomization hash. Placeholders in the companion JSON are not launch
authority.

`frozen.command_hash` is the hash of the canonical campaign command template,
not a claim that all 36 expanded argvs are identical. The bound descendant
must additionally contain a complete immutable `cell_command_hashes` object
mapping every expected `cell_id` to the SHA-256 of its exact argv. Each result
must match its cell-specific hash. Missing, extra, or duplicate map keys fail
before launch.

## 3. Locked evaluation

The primary outcome is final validation loss per target token on exactly 1,024
fixed, disjoint examples. The exact example IDs, order, tokenized packed
sequences, attention masks, labels, loss sums, and target-token counts are
hashed before any scientific launch and reused for every arm and seed.

Requirements:

- retain the aggregate loss at at least 12 significant digits;
- retain per-sequence loss sum and target-token count for paired evaluation;
- do not regenerate the evaluation split after a failure or preemption;
- report paired-example uncertainty as evaluation uncertainty only;
- never call evaluation-row uncertainty training-seed uncertainty;
- never reuse `0.009` as a noise estimate.

Final endpoint NLL is primary. Training loss, curve AUC, tape geometry, rho,
aligned gain, and transverse energy are secondary diagnostics and cannot
replace a failed endpoint gate.

## 4. P0: loss-blind semantic canary

P0 uses shuffle/training seed `337 / 337337` on one Spot A100. It is explicitly
**non-evidence** and never enters an effect estimate, LR choice, or paper table.
It may use abbreviated work solely to verify:

1. all four learners participate in every full-quorum commit;
2. barrier execution prevents a learner from starting the next window before
   the registered broadcast;
3. every admitted push is differenced against its recorded base version;
4. fixed-window `c_steps=H` and `c_tokens=128H` hold for every responder;
5. RDA replay and the Nesterov recurrence reproduce the applied step;
6. the resolved merge mode, fragment layout, flags, hashes, and work counters
   appear in sealed artifacts;
7. no injected historical baseline is present;
8. per-sequence evaluation sums reproduce the aggregate loss;
9. the Spot VM, GPU, disk, image, command, and teardown evidence are complete.

P0 losses remain quarantined. Any P0 semantic failure blocks P1 until fixed,
followed by a new P0 canary and a new immutable manifest. Fixes cannot be made
silently under the old preregistration hash.

## 5. P1-R0: initial low-LR grid

P1-R0 uses the fresh development shuffle/training seed `347 / 347347`.
Repository search found no prior use of this seed family at freeze time.

The exact Cartesian grid is:

```text
H   = {16, 64, 256}
mu  = {0, .5, .9}
eta = {.021875, .04375, .0875, .175}
```

This is 36 expected cells. Eta=.175 is mandatory: it is the new-protocol upper
neighbor needed to avoid declaring eta=.0875 bracketed merely because legacy
EXP2.30 sampled it. Every `(H, eta, seed)` block contains a live mu=0 arm and
the two momentum arms. No `--baseline-loss` or imported absolute loss is
permitted.

## 6. Sequential bracketing rule

Adaptation is allowed only on the P1 development seed and only through this
algorithm. Each adaptive round receives a new immutable expected-grid manifest
before launch; the P1-R0 manifest is never edited.

For every `(H, mu)` independently:

1. After all 36 P1-R0 cells are valid and sealed, find the eta with lowest
   high-precision locked-eval NLL.
2. If eta=.021875 is best, add eta=.0109375. If the new lower boundary is best,
   continue halving through `.00546875`, `.002734375`, and `.0013671875`, one
   sealed round at a time.
3. If the best eta remains the lower boundary after `.0013671875`, label that
   `(H, mu)` **UNBRACKETED-LOW** and fail the tuned-LR gate. Do not extrapolate
   an optimum or open later seeds.
4. If eta=.175 is best, add eta=.35, then eta=.7 if needed. If eta=.7 remains
   best, label the cell **UNBRACKETED-HIGH** and fail the gate.
5. Once an interior point is best, add the geometric midpoint between it and
   each adjacent sampled neighbor exactly once. The final development choice
   is the lowest point estimate on this refined grid; all adjacent losses and
   paired-example intervals are reported.
6. Divergent hyperparameters remain scientific outcomes and are assigned
   infinite loss for tuning. Infrastructure failures remain missing until a
   loss-blind allowed retry succeeds.

No legacy loss enters this algorithm. No seed 359, 373, or confirmation seed
may be evaluated or unblinded while P1 bracketing is in progress.

## 7. Randomization, Spot preemption, and retries

P1 runs on Spot capacity only. There is no on-demand fallback. Scientific
blocks are `(H, eta, seed)` triples containing mu=0/.5/.9. The 12 initial
blocks are put in a committed pseudorandom order; momentum order is independently
permuted within each block. The randomization plan is materialized and hashed
before launch.

This blocking keeps every momentum result close in wall time to its live mu=0
control. Hardware identity, zone, start/end time, attempt, and order index are
recorded so residual time-block or preemption effects are auditable.

Allowed retry triggers are loss-blind and mechanical only:

- provider-reported Spot preemption;
- VM/host/GPU failure;
- process exit before a scientific divergence is recorded;
- missing or checksum-invalid required artifact;
- validator-detected command/provenance mismatch before outcomes are opened.

An incomplete/preempted block is marked infrastructure failure and the entire
three-arm block is rerun from identical initial state, seed, command, image,
and data on the same machine/GPU class. All attempts are retained and linked by
`retry_of`. The retry decision must be logged before aggregate or per-example
loss is inspected. A finite completed loss, a poor loss, or a preregistered
hyperparameter divergence is never a retry reason.

## 8. P1 go/kill decisions

The following numerical margins are prospective resource/importance gates,
not claims about a measured noise floor.

P1 may advance to the three-development-seed stage only if:

1. all semantic, provenance, coverage, evaluation, and live-control validators
   pass;
2. all nine `(H, mu)` LR curves are bracketed under Section 6;
3. `D_short >= +0.020`; and
4. `D_long <= -0.010`.

Interpretation of failures is frozen:

| outcome | required conclusion |
|---|---|
| D_short < +.020 after tuning | the headline short-H effect is substantially an LR-matching effect in this full-parameter setting; kill the claimed tuned poison |
| D_short passes, D_long >= 0 | high momentum can be harmful, but there is no tuned beneficial crossover; kill the phase-transition/controller framing |
| -.010 < D_long < 0 | at most a weak exploratory long-H benefit; insufficient for the best-paper gate |
| any LR curve unbracketed | no tuned comparison is allowed; extend only as Section 6 permits, otherwise stop |
| missing/invalid live control | affected block is inadmissible, not evidence |

Regardless of sign or size, P1 alone cannot corroborate robustness.

## 9. Advancement to three and eight seeds

### P2: three-seed development

If P1 passes, add unopened development seeds:

```text
shuffle/training = 359/359359 and 373/373373
```

For each `(H, mu)`, freeze the P1-selected eta and its immediate lower and
higher bracket neighbors before launching either new seed. Run all expected
cells on both seeds, retain live mu=0 controls, and unblind seeds 359 and 373
only after both complete and validate. P1 seed 347 plus these two seeds form a
three-seed **development** set.

Advance only if the short and long contrasts have their registered signs in at
least two of three seeds and their three-seed means still satisfy
`D_short >= +.020` and `D_long <= -.010`. This is a promotion gate, not final
inference. Any formula, controller, selected momentum, LR, contrast, tolerance,
or analysis changed after P2 requires a completely new confirmation seed set.

### P3: eight fresh confirmation seeds

The untouched confirmation shuffle seeds are:

```text
383, 397, 409, 421, 433, 443, 457, 461
```

Their corresponding training seeds are the duplicated decimal forms
`383383`, ..., `461461`. They are not run, evaluated, or unblinded during P1 or
P2.

Before P3, freeze one short-H pair and one long-H pair from P2, their exact
etas, all code/config/hashes, and a complete expected-grid manifest. Each pair
includes a fresh live mu=0 arm for every seed. Run all eight seeds; do not use
sequential stopping and do not open partial outcomes.

The two co-primary seed-level paired contrasts use Holm-adjusted two-sided
95% confidence intervals. Confirmation requires:

- short-H mean penalty at least +.020 and adjusted lower endpoint above zero;
- long-H mean benefit at most -.010 and adjusted upper endpoint below zero;
- no provenance, coverage, semantic, or live-control failure.

A failed co-primary contrast is reported as a failure. Evaluation-example
bootstraps may accompany but cannot replace the eight training-seed pairs.

## 10. SNOO and Outer-Momentum Restarting reconciliation

P1 is not a direct replication of either neighboring result, and the paper
must not call them contradictory.

- SNOO (`arXiv:2510.15830`) reports beneficial pseudo-gradient Nesterov in an
  M=1, jointly tuned, long-run C4 pretraining regime over step frequencies up
  to 400 and model sizes from roughly 125M to 1B, with much larger token
  budgets. M=1 removes cross-worker averaging entirely.
- Outer-Momentum Restarting (`arXiv:2605.28585`) uses Llama-150M on C4,
  sequence length 2,048, M=2, about 3.3B tokens, H in
  `{64,128,512,1024,2048}`, outer LR in `{.1,.3,.5,.7,.9,1.1}`, and momentum
  in `{.1,.3,.5,.7,.9}`. Its main claim is restart robustness across the
  hyperparameter region, not that every no-restart momentum arm is optimal.

These regimes differ from legacy EXP2.30 in worker count, data/objective,
sequence length, token budget, H range, LR range, synchronization semantics,
and tuning protocol. A universal paper claim therefore requires two later
reconciliation blocks after P3:

1. **M-axis bridge:** repeat the frozen tuned phase-map comparison at M=1 and
   M=4 on the same model/data, measuring the full temporal kernel and realized
   buffer geometry. This tests whether SNOO's benefit follows the M-induced
   temporal regime rather than contradicting it.
2. **OMR benchmark bridge:** run a faithful C4/Llama-150M, M=2 long-horizon
   block covering at least H64 and H2048, with tuned no-restart Nesterov and
   the authors' hard-restart baseline under equal tuning budgets. The full
   `{64,...,2048}` axis is required before claiming horizon-wide superiority or
   that a new controller subsumes restarting.

Until those blocks run, conclusions are scoped to the frozen SmolLM2/Capybara
M=4 setting.

## 11. Required evidence bundle

Every expected arm, including failures and divergences, must have:

- immutable study/round/cell ID and exact command;
- canonical campaign-command hash plus a complete expected-cell-to-command-
  hash map bound before launch;
- H, mu, eta, shuffle seed, training seed, attempt, block, and order index;
- clean Git commit and image/model/data/eval/command/randomization hashes;
- machine type, GPU identity, Spot status, zone, VM/disk IDs and nonce;
- barrier, version-match, strict-quorum, RDA, correction, dtype, and delay flags;
- target and observed tokens, microsteps, global commits, and per-fragment
  update counts;
- start/end timestamps and infrastructure/scientific status;
- high-precision endpoint and per-sequence evaluation artifact;
- tape/capture URI and SHA-256;
- paired live-control cell ID;
- retry lineage and mechanical retry reason, if any;
- GPU teardown and zero-accelerator proof after artifact sealing.

Unexpected, duplicate, missing, silently retried, or hash-mismatched cells fail
validation. Infra failures are not scientific outcomes; scientific divergences
are not infra failures.
