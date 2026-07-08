# EXP2: Equal-Token Late Fragment Utility

Date: 2026-07-07

## Question

In the real learner/syncer path, with a real Hugging Face model and dataset, do fixed-token fragment snapshots still have materially different one-step utility after artificial arrival delay?

EXP1 showed that the first real HF stress run had a confound: token counters varied substantially, and token count became predictive. EXP2 changes the stress protocol so delay is applied after the learner has generated a fixed-window fragment snapshot.

## Assets

Primary debug assets:

| Item | Value |
|---|---|
| Model | `HuggingFaceTB/SmolLM2-135M` |
| Dataset | `trl-lib/Capybara` |
| Logical learners | `12` |
| Fragments | `12` |
| Quorum | `6` |
| Probe data | held-out eval split from the same shuffled dataset |

The data is not generated synthetically. `scripts/compare_diloco.py` materializes the Hugging Face rows into `train.jsonl` and `eval.jsonl`, with optional deterministic row shuffling through `--shuffle-rows-seed`.

## Implemented Controls

New learner flags:

```bash
--fixed-window-tokens <N>
--fixed-window-microsteps <N>
--pad-to-fixed-window-tokens
--freeze-delta-before-delay
```

Semantics:

- the learner trains until a fragment has crossed the fixed local window after its last reset,
- rank 0 snapshots the flat fragment tensor, local step, base version, local-step count, and token count,
- later pull responses for that fragment are packed from the cached snapshot,
- artificial push delay is applied after the cached payload/probe candidate is materialized,
- the snapshot is cleared only when a broadcast for that fragment resets the fragment window.

This isolates arrival delay from local token accumulation. In a clean run, all candidates for a fragment round should have nearly identical `c_tokens`.

## Probe Status

Current implemented oracle scopes:

```text
learner_known_global
syncer_current_global
```

The learner-known probe evaluates the candidate against the last broadcast anchors known to that learner. This is enough to debug the equal-token stress path. The syncer-current probe is now implemented as offline replay from captured pre-merge syncer checkpoints and admitted candidate fragments.

Still pending before end-to-end policy runs:

- stale-replay depth mode,
- a calibrated score with stable transfer across learners, time, and seeds.

Update after instrumentation:

- the syncer can now capture admitted candidate fragments plus the pre-merge syncer state,
- an offline evaluator can score captured candidates against the syncer-current state,
- the existing three p4de seeds remain learner-known-oracle runs because candidate tensors and pre-merge syncer checkpoints were not captured at the time.

Probe logging now includes:

- aggregate utility,
- `utility_se` when at least two probe batches are used,
- `bad_strict`, defined as `utility + utility_se < 0`.

## Main Debug Command

```bash
python scripts/compare_diloco.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --data trl-lib/Capybara \
  --settings m12 \
  --token-budget 200000 \
  --seq-len 128 \
  --micro-batch-size 1 \
  --inner-lr 0.001 \
  --lora-r 2 \
  --lora-alpha 4 \
  --eval-rows 256 \
  --max-rows 5000 \
  --shuffle-rows-seed 17 \
  --device cuda \
  --gpu-slots 12 \
  --probe-data eval \
  --probe-every 1 \
  --probe-batches 8 \
  --probe-batch-size 1 \
  --probe-max-rows 256 \
  --fixed-window-tokens 8192 \
  --fixed-window-microsteps 64 \
  --pad-to-fixed-window-tokens \
  --freeze-delta-before-delay \
  --learner-push-delay-ms 0,50,100,150,200,250,0,75,150,225,300,375 \
  --learner-delay-jitter-ms 20 \
  --work-dir experiment-results/EXP2/equal-token-late-smollm2 \
  --report-dir experiment-results/EXP2/equal-token-late-smollm2
```

Summarize probe logs:

```bash
python scripts/summarize_fragment_probe.py \
  experiment-results/EXP2/equal-token-late-smollm2/m12/fragment_probe_learner_*.jsonl \
  --out experiment-results/EXP2/equal-token-late-smollm2/probe_summary.json
```

## Summary Fields

`scripts/summarize_fragment_probe.py` now emits the fields needed for the EXP2 gate:

| Field | Meaning |
|---|---|
| `records` | probe records loaded |
| `round_token_cv_mean` | mean within-round token-count CV grouped by `(pull_step, fragment)` |
| `round_token_cv_p95` | p95 within-round token-count CV |
| `utility_noise_estimate` | mean `utility_se` over records with enough probe batches |
| `negative_utility_rate` | fraction with `utility < 0` |
| `bad_strict_rate` | fraction with `utility + utility_se < 0` |
| `token_auroc` | bad-fragment AUROC from token count |
| `freshness_auroc` | bad-fragment AUROC from freshness |
| `alignment_auroc` | bad-fragment AUROC from alignment |
| `hand_score_auroc` | bad-fragment AUROC from the current hand score |
| `calibrated_score_auroc` | bad-fragment AUROC from a trained diagnostic score when records include `calibrated_score` |
| `score_minus_token_auroc` | hand score AUROC minus token AUROC |

## Go Conditions

Measurement cleanliness:

- `round_token_cv_mean < 0.03`,
- `round_token_cv_p95 < 0.05`,
- at least `5,000` probe records for main analysis,
- at least `3` seeds for the real result.

Phenomenon:

- `negative_utility_rate >= 0.10`,
- utility `p95 - p05` clearly exceeds the probe-noise estimate,
- token-count bad-fragment AUROC stays near chance.

Signal:

- the best simple score reaches useful separation, ideally AUROC `>= 0.65`,
- score AUROC beats token AUROC by at least `0.10`,
- if this fails but utility spread is real, scoring needs redesign before any end-to-end method run.

## Current Interpretation

EXP2 is now instrumented for the equal-token delay path, and the real HF learner/syncer path has been exercised on both a one-GPU diagnostic and several 8xA100 p4de runs.

The learner-known three-seed p4de replication passes the measurement gate:

- token-count variation is exactly zero under fixed-window snapshots,
- token count is chance-level for bad-fragment prediction,
- negative-utility fragments remain common,
- utility spread is larger than the probe-noise estimate.

The syncer-current three-seed replication is stronger for the measurement claim:

- token-count variation is still exactly zero,
- token AUROC remains exactly `0.500`,
- negative-utility fragments average `44.2%`,
- strict bad fragments average `18.0%`.

The sampled merge-utility replay adds a first actual-merge check:

- token-weighted merged updates are negative in `39.6%` of sampled captured groups,
- strict-negative merged updates occur in `20.8%` of sampled captured groups,
- oracle-positive filtering improves mean one-step merge utility by `0.00159` over token-weighted merging.

The current hand score does not pass the signal gate. Across learner-known seeds, its AUROC is effectively chance-level and only `0.0081` above token count on average. Across syncer-current seeds, its AUROC is below token count on average at `0.494`.

The anchor-gradient diagnostic changes the scoring picture. A raw current-state gradient-dot feature averages `0.707` AUROC across seeds, and a calibrated anchor-gradient score trained on two seeds and tested on the held-out seed averages `0.742` AUROC with calibration error about `0.100`. These runs support the equal-token measurement phenomenon, show sampled merge headroom, and identify a stronger current-state utility signal. The next required work is offline policy replay with anchor-gradient ranking, then stale-replay depth mode.

## Syncer-Current Capture Path

The syncer now supports offline candidate capture:

```bash
--probe-capture-dir <arm-dir>/syncer_probe
--probe-capture-every <N>
```

When enabled, each sampled completed round writes:

```text
syncer_probe/index.jsonl
syncer_probe/states/state_before_step_XXXXXXXX.ckpt
syncer_probe/candidates/candidate_step_XXXXXXXX_fragment_XXXX_learner_XXXX.f32
```

The checkpoint is saved immediately before the outer merge for that round. Candidate `.f32` files contain the admitted learner fragment values. This is enough for an offline evaluator to compute utility against the syncer's current global model:

```text
utility = L(theta_syncer_current) - L(theta_syncer_current with fragment trial)
trial_fragment = current_fragment + eta * (candidate_fragment - current_fragment)
```

Comparison runs enable capture with:

```bash
python scripts/compare_diloco.py \
  ... \
  --syncer-probe-capture \
  --syncer-probe-capture-every 1
```

Offline evaluation:

```bash
python scripts/evaluate_syncer_probe_capture.py \
  --capture-dir <arm-dir>/syncer_probe \
  --model HuggingFaceTB/SmolLM2-135M \
  --data <work-dir>/eval.jsonl \
  --seq-len 128 \
  --device cuda \
  --lora-r 2 \
  --lora-alpha 4 \
  --fragments 12 \
  --probe-batches 8 \
  --probe-batch-size 1 \
  --probe-max-rows 256 \
  --max-records 2048
```

The evaluator writes:

```text
syncer_probe/syncer_current_probe.jsonl
syncer_probe/syncer_current_probe_summary.json
```

The emitted records use the same `fragment_probe_v2` schema as learner-known probes, but set:

```text
oracle_scope = syncer_current_global
```

Tiny local smoke:

| Item | Value |
|---|---:|
| Model | `/tmp/yeto-exp1-real/tiny-llama` |
| Data | `/tmp/yeto-exp1-real/chat.jsonl` |
| Arm | `m2` |
| Token budget | 2,048 |
| Captured candidates | 122 |
| Offline sample records | 8 |

Smoke artifacts:

```text
/tmp/yeto-syncer-probe-smoke/work/m2/syncer_probe/index.jsonl
/tmp/yeto-syncer-probe-smoke/work/m2/syncer_probe/syncer_current_probe_sample.jsonl
/tmp/yeto-syncer-probe-smoke/work/m2/syncer_probe/syncer_current_probe_sample_summary.json
```

This smoke only verifies that syncer-state capture, candidate tensor capture, offline model loading, checkpoint application, trial-fragment evaluation, and summary emission work end to end. It is not evidence for the EXP2 phenomenon.

## Real HF P4DE Syncer-Current Seed-53 Run

This run repeats the equal-token late-fragment setup and adds syncer-current capture every 4 outer steps.

Common setting:

| Item | Value |
|---|---:|
| Region / instance | `us-east-1d` / `p4de.24xlarge` spot |
| GPU | 8x A100 80GB |
| Model | `HuggingFaceTB/SmolLM2-135M` |
| Dataset | `trl-lib/Capybara` |
| Shuffle seed | 53 |
| Logical learners | 12 |
| GPU slots | 8 |
| Fragments | 12 |
| Quorum | 6 |
| Token budget | 6,200,000 |
| Steps per learner | 4,037 |
| Fixed window | 8,192 tokens / 64 microsteps |
| Probe batches | 8 |
| Syncer capture cadence | every 4 outer steps |
| Captured syncer candidates | 1,158 |
| Syncer-current evaluated sample | 512 |
| Raw syncer capture size | 1.3 GB |

Learner-known probe summary:

| Metric | Value |
|---|---:|
| Records | 8,676 |
| `round_token_cv_mean` | 0.000 |
| `round_token_cv_p95` | 0.000 |
| `negative_utility_rate` | 0.389 |
| `bad_strict_rate` | 0.153 |
| `utility_noise_estimate` | 0.00299 |
| utility p05 | -0.00510 |
| utility p50 | 0.00105 |
| utility p95 | 0.00812 |
| `token_auroc` | 0.500 |
| `freshness_auroc` | 0.528 |
| `alignment_auroc` | 0.504 |
| `hand_score_auroc` | 0.499 |
| `score_minus_token_auroc` | -0.0015 |

Syncer-current sampled probe summary:

| Metric | Value |
|---|---:|
| Records | 512 |
| `round_token_cv_mean` | 0.000 |
| `round_token_cv_p95` | 0.000 |
| `negative_utility_rate` | 0.504 |
| `bad_strict_rate` | 0.232 |
| `utility_noise_estimate` | 0.00300 |
| utility p05 | -0.00564 |
| utility p50 | -0.000007 |
| utility p95 | 0.00666 |
| `token_auroc` | 0.500 |
| `freshness_auroc` | 0.506 |
| `alignment_auroc` | 0.489 |
| `hand_score_auroc` | 0.480 |
| `score_minus_token_auroc` | -0.0202 |

Artifacts:

```text
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m-light.tgz
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m.tgz
```

Interpretation:

- The learner-known result replicates the three-seed p4de pattern: token count is exactly controlled, token AUROC is 0.500, utility spread remains large, and the current hand score is chance-level.
- The syncer-current sample is stronger and harsher than the learner-known oracle: negative-utility rate rises from 38.9% to 50.4%, and strict bad rate rises from 15.3% to 23.2%.
- Token count remains controlled away under the syncer-current oracle: token AUROC is 0.500.
- Current hand scoring is not useful under the syncer-current oracle: `hand_score_auroc` is 0.480.
- This supports the measurement direction but rules out using the present hand score for an end-to-end policy. The replicated syncer-current diagnostics below test whether calibration transfers across learners, time, and seeds.

## Calibrated Syncer-Current Score: Seed-53 Diagnostic

This diagnostic trains a lightweight logistic score on the 512 syncer-current records from the seed-53 run. It is intentionally small and should be read as a scoring diagnostic, not a main result.

Source:

```text
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/syncer_current_probe_512.jsonl
```

Features:

```text
age, freshness, alignment, uncertainty, norm_anomaly, log(update_norm), c_steps, c_tokens, optional fragment one-hot
```

Primary split:

```text
train: learners 0-7
test: learners 8-11
```

Diagnostic results:

| Variant | Split | Train records | Test records | Test negative utility | Token AUROC | Hand score AUROC | Calibrated AUROC | Calibrated ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fragment-aware | held-out learners | 390 | 122 | 0.467 | 0.500 | 0.462 | 0.570 | 0.102 |
| no fragment one-hot | held-out learners | 390 | 122 | 0.467 | 0.500 | 0.462 | 0.476 | 0.088 |
| fragment-aware | late rounds | 384 | 128 | 0.477 | 0.500 | 0.477 | 0.454 | 0.179 |
| strict-label train | held-out learners | 390 | 122 | 0.467 | 0.500 | 0.462 | 0.572 | 0.235 |

Artifacts:

```text
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/calibrated_syncer_current_heldout_learners/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/calibrated_syncer_current_heldout_learners_no_fragment/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/calibrated_syncer_current_late_rounds/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/calibrated_syncer_current_heldout_learners_strict/
```

Interpretation:

- The fragment-aware held-out learner split improves over token count and the hand score, but only modestly: AUROC rises to about 0.57, below the `>= 0.65` target.
- Removing fragment identity drops the calibrated score below chance on held-out learners. This suggests the current feature set is not yet learning a portable utility signal.
- Late-round transfer is also below chance, despite strong train AUROC. That is a warning sign for overfit or nonstationarity.
- Strict-label training gives similar AUROC but much worse calibration error.
- For metadata-only scoring, the scoring gate remains open. Later diagnostics below test a current-state anchor-gradient signal.

## Real HF P4DE Syncer-Current Replication

These runs repeat the equal-token late-fragment setup and evaluate captured candidate fragments against the syncer's pre-merge current state.

Common setting:

| Item | Value |
|---|---:|
| Region / instance | `us-east-1d` / `p4de.24xlarge` spot |
| GPU | 8x A100 80GB |
| Model | `HuggingFaceTB/SmolLM2-135M` |
| Dataset | `trl-lib/Capybara` |
| Logical learners | 12 |
| GPU slots | 8 |
| Fragments | 12 |
| Quorum | 6 |
| Token budget | 6,200,000 |
| Fixed window | 8,192 tokens / 64 microsteps |
| Probe batches | 8 |
| Syncer capture cadence | every 4 outer steps |

Learner-known probe replication for the same three runs:

| Seed | Records | Token CV mean | Token CV p95 | Negative utility | Bad strict | Token AUROC | Freshness AUROC | Alignment AUROC | Hand score AUROC | Score - token | Utility p05 | Utility p50 | Utility p95 | Noise est. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 53 | 8,676 | 0.000 | 0.000 | 0.389 | 0.153 | 0.500 | 0.528 | 0.504 | 0.499 | -0.0015 | -0.00510 | 0.00105 | 0.00812 | 0.00299 |
| 67 | 8,664 | 0.000 | 0.000 | 0.380 | 0.130 | 0.500 | 0.537 | 0.475 | 0.512 | 0.0116 | -0.00477 | 0.00110 | 0.00834 | 0.00389 |
| 79 | 8,786 | 0.000 | 0.000 | 0.366 | 0.127 | 0.500 | 0.543 | 0.470 | 0.514 | 0.0143 | -0.00406 | 0.00111 | 0.00766 | 0.00284 |

Learner-known aggregate:

| Metric | Mean | Std |
|---|---:|---:|
| Records | 8,708.7 | 67.2 |
| `negative_utility_rate` | 0.379 | 0.012 |
| `bad_strict_rate` | 0.137 | 0.014 |
| `token_auroc` | 0.500 | 0.000 |
| `freshness_auroc` | 0.536 | 0.008 |
| `alignment_auroc` | 0.483 | 0.018 |
| `hand_score_auroc` | 0.508 | 0.008 |
| `score_minus_token_auroc` | 0.0081 | 0.0084 |
| `utility_noise_estimate` | 0.00324 | 0.00057 |
| utility p05 | -0.00464 | 0.00053 |
| utility p50 | 0.00109 | 0.00003 |
| utility p95 | 0.00804 | 0.00035 |

Syncer-current sampled probe replication:

| Seed | Records | Token CV mean | Token CV p95 | Negative utility | Bad strict | Token AUROC | Freshness AUROC | Alignment AUROC | Hand score AUROC | Score - token | Utility p05 | Utility p50 | Utility p95 | Noise est. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 53 | 512 | 0.000 | 0.000 | 0.504 | 0.232 | 0.500 | 0.506 | 0.489 | 0.480 | -0.0202 | -0.00564 | -0.00001 | 0.00666 | 0.00300 |
| 67 | 1,024 | 0.000 | 0.000 | 0.435 | 0.147 | 0.500 | 0.511 | 0.485 | 0.484 | -0.0162 | -0.00536 | 0.00057 | 0.00896 | 0.00423 |
| 79 | 1,024 | 0.000 | 0.000 | 0.389 | 0.159 | 0.500 | 0.511 | 0.510 | 0.519 | 0.0193 | -0.00453 | 0.00107 | 0.00837 | 0.00296 |

Syncer-current aggregate:

| Metric | Mean | Std |
|---|---:|---:|
| Records | 853.3 | 295.6 |
| `negative_utility_rate` | 0.442 | 0.058 |
| `bad_strict_rate` | 0.180 | 0.046 |
| `token_auroc` | 0.500 | 0.000 |
| `freshness_auroc` | 0.509 | 0.003 |
| `alignment_auroc` | 0.495 | 0.013 |
| `hand_score_auroc` | 0.494 | 0.022 |
| `score_minus_token_auroc` | -0.0057 | 0.0218 |
| `utility_noise_estimate` | 0.00340 | 0.00073 |
| utility p05 | -0.00518 | 0.00057 |
| utility p50 | 0.00054 | 0.00054 |
| utility p95 | 0.00799 | 0.00120 |

Calibrated syncer-current diagnostics:

| Split | Seed | Train records | Test records | Test negative utility | Token AUROC | Hand score AUROC | Calibrated AUROC | Calibrated ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| held-out learners | 53 | 390 | 122 | 0.467 | 0.500 | 0.462 | 0.570 | 0.102 |
| held-out learners | 67 | 834 | 190 | 0.442 | 0.500 | 0.452 | 0.603 | 0.091 |
| held-out learners | 79 | 829 | 195 | 0.349 | 0.500 | 0.611 | 0.644 | 0.083 |
| late rounds | 53 | 384 | 128 | 0.477 | 0.500 | 0.477 | 0.454 | 0.179 |
| late rounds | 67 | 768 | 256 | 0.500 | 0.500 | 0.489 | 0.438 | 0.167 |
| late rounds | 79 | 768 | 256 | 0.383 | 0.500 | 0.533 | 0.657 | 0.070 |

Calibration aggregate:

| Split | Metric | Mean | Std |
|---|---|---:|---:|
| held-out learners | test negative utility | 0.419 | 0.062 |
| held-out learners | token AUROC | 0.500 | 0.000 |
| held-out learners | hand score AUROC | 0.508 | 0.089 |
| held-out learners | calibrated AUROC | 0.606 | 0.037 |
| held-out learners | calibrated ECE | 0.092 | 0.010 |
| late rounds | test negative utility | 0.453 | 0.062 |
| late rounds | token AUROC | 0.500 | 0.000 |
| late rounds | hand score AUROC | 0.500 | 0.030 |
| late rounds | calibrated AUROC | 0.516 | 0.122 |
| late rounds | calibrated ECE | 0.139 | 0.060 |

Artifacts:

```text
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m-light.tgz
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m.tgz
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed67-syncer-current-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed67-syncer-current-6m-light.tgz
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed67-syncer-current-6m.tgz
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed79-syncer-current-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed79-syncer-current-6m-light.tgz
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed79-syncer-current-6m.tgz
```

Interpretation:

- The syncer-current oracle preserves the clean measurement controls: token CV is exactly zero and token count remains chance-level.
- Harmful fragments remain common under the stricter syncer-current oracle: negative utility averages 44.2% and strict bad rate averages 18.0%.
- Utility spread is larger than probe noise: syncer-current p95-p05 averages about 0.0132, while the noise estimate averages about 0.0034.
- Freshness, alignment, and the hand score are all near chance under the syncer-current oracle.
- The metadata-only calibrated score is directionally useful on held-out learners but not stable over time. The late-round split succeeds on seed 79 and fails on seeds 53 and 67.
- The measurement gate passes. Metadata-only scoring does not pass. The later EXP2.5B anchor-gradient diagnostic is the first candidate-level scoring result that transfers across held-out seeds.

## EXP2.5A Sampled Merge-Utility Replay

This replay asks whether harmful individual candidates are averaged away or whether they leave measurable headroom in the actual merged fragment update.

Script:

```text
scripts/replay_merge_utility.py
```

Replay setting:

| Item | Value |
|---|---:|
| Seeds | 53, 67, 79 |
| Groups per seed | 64 sampled complete `(step, fragment)` groups |
| Minimum candidates per group | 2 |
| Probe batches | 4 |
| Probe batch size | 1 |
| Probe max rows | 128 |
| Device | local CPU |
| Model/data | same as syncer-current runs |

Policies:

| Policy | Definition |
|---|---|
| token-weighted | captured candidates merged with their syncer token weights |
| freshness-weighted | token weights multiplied by candidate freshness |
| hand-score-weighted | token weights multiplied by the current hand score |
| oracle-positive | token-weighted merge of candidates with positive individual utility |
| oracle-topk | token-weighted merge of top 50% candidates by individual utility |
| oracle-drop-strict-bad | token-weighted merge after removing candidates with `utility + utility_se < 0` |
| random controls | random candidate subsets with the same selected counts as oracle-positive and oracle-drop-strict-bad |

Per-seed summary:

| Seed | Groups | Candidate count | Bad weight mass | Strict-bad weight mass | Token utility | Token negative | Token strict neg. | Oracle-positive utility | Oracle-positive headroom | Oracle-topk utility | Oracle-topk headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 53 | 64 | 5.80 | 0.484 | 0.238 | 0.00072 | 0.375 | 0.203 | 0.00258 | 0.00186 | 0.00231 | 0.00159 |
| 67 | 64 | 5.66 | 0.499 | 0.201 | 0.00142 | 0.422 | 0.188 | 0.00300 | 0.00158 | 0.00301 | 0.00159 |
| 79 | 64 | 5.59 | 0.434 | 0.229 | 0.00191 | 0.391 | 0.234 | 0.00324 | 0.00133 | 0.00322 | 0.00131 |

Aggregate:

| Metric | Mean | Std |
|---|---:|---:|
| groups per seed | 64.0 | 0.0 |
| candidate count | 5.68 | 0.10 |
| bad weight mass | 0.472 | 0.034 |
| strict-bad weight mass | 0.223 | 0.019 |
| token-weighted utility | 0.00135 | 0.00060 |
| token-weighted negative rate | 0.396 | 0.024 |
| token-weighted strict-negative rate | 0.208 | 0.024 |
| freshness-weighted utility | 0.00135 | 0.00060 |
| hand-score-weighted utility | 0.00135 | 0.00062 |
| oracle-positive utility | 0.00294 | 0.00033 |
| oracle-positive headroom | 0.00159 | 0.00027 |
| oracle-positive headroom positive rate | 0.792 | 0.063 |
| oracle-topk utility | 0.00285 | 0.00047 |
| oracle-topk headroom | 0.00150 | 0.00016 |
| oracle-topk headroom positive rate | 0.984 | 0.016 |
| oracle-drop-strict-bad utility | 0.00211 | 0.00055 |
| oracle-drop-strict-bad headroom | 0.00077 | 0.00010 |
| random-positive-count utility | 0.00172 | 0.00042 |
| random-drop-strict-bad-count utility | 0.00126 | 0.00066 |

Artifacts:

```text
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/merge_utility_replay_64.jsonl
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/merge_utility_summary_64.json
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed67-syncer-current-6m/merge_utility_replay_64.jsonl
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed67-syncer-current-6m/merge_utility_summary_64.json
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed79-syncer-current-6m/merge_utility_replay_64.jsonl
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed79-syncer-current-6m/merge_utility_summary_64.json
```

Interpretation:

- Bad individual candidates are not fully averaged away. In the sampled replay, token-weighted merged updates are negative in 39.6% of groups and strict-negative in 20.8%.
- Oracle-positive and oracle-topk policies expose measurable headroom over token-weighted merging. The mean headroom is about the same size as the token-weighted mean utility itself.
- Freshness-weighted and hand-score-weighted merging are effectively indistinguishable from token-weighted merging in this replay.
- Random-count controls improve less than oracle-positive and oracle-topk, so the headroom is not just from using fewer candidates.
- This is still a sampled replay, not a full-capture replay. The merge-headroom gate is promising, but the next step should run the full replay on GPU or add checkpointed progress output before scaling locally.

## EXP2.5B Anchor-Gradient Candidate Scoring

This diagnostic replaces metadata-only scoring with a current-state anchor-gradient feature computed at the syncer checkpoint:

```text
probe_grad_dot = < -grad_anchor(theta_current)_fragment, candidate_fragment - current_fragment >
```

For each syncer-current candidate record, the evaluator:

1. loads the captured pre-merge syncer checkpoint,
2. computes an anchor loss gradient on fixed probe batches,
3. extracts the fragment gradient,
4. computes gradient-dot, gradient-cosine, normalized-dot, curvature-penalized-dot, and consensus features,
5. writes the enriched candidate record with the original utility label.

Scripts:

```text
scripts/evaluate_anchor_gradient_features.py
scripts/calibrate_fragment_score.py --split heldout-seed
```

Feature extraction setting:

| Item | Value |
|---|---:|
| Seeds | 53, 67, 79 |
| Records | 512 / 1,024 / 1,024 |
| Anchor batches | 2 |
| Anchor batch size | 1 |
| Anchor max rows | 64 |
| Device | local CPU |
| Model/data | same as syncer-current runs |

Raw feature quality:

| Seed | Records | Bad rate | `probe_grad_dot` AUROC | `probe_grad_dot` Pearson | `probe_grad_dot` Spearman | `probe_grad_cosine` AUROC | Hand score AUROC | Freshness AUROC | Alignment AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 53 | 512 | 0.504 | 0.686 | 0.476 | 0.419 | 0.694 | 0.480 | 0.506 | 0.489 |
| 67 | 1,024 | 0.435 | 0.706 | 0.618 | 0.452 | 0.695 | 0.484 | 0.511 | 0.485 |
| 79 | 1,024 | 0.389 | 0.729 | 0.595 | 0.500 | 0.716 | 0.519 | 0.511 | 0.510 |

Raw feature aggregate:

| Metric | Mean | Std |
|---|---:|---:|
| `probe_grad_dot` AUROC | 0.707 | 0.021 |
| `probe_grad_dot` Pearson | 0.563 | 0.076 |
| `probe_grad_dot` Spearman | 0.457 | 0.041 |
| `probe_grad_cosine` AUROC | 0.702 | 0.013 |
| `combined_score` AUROC | 0.494 | 0.022 |
| `freshness` AUROC | 0.509 | 0.003 |
| `alignment` AUROC | 0.495 | 0.013 |

Held-out-seed calibration:

| Test seed | Train seeds | Train records | Test records | Test bad rate | Token AUROC | Hand score AUROC | Calibrated AUROC | Calibrated ECE | Calibrated Pearson | Calibrated Spearman |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 53 | 67, 79 | 2,048 | 512 | 0.504 | 0.500 | 0.480 | 0.733 | 0.095 | 0.560 | 0.528 |
| 67 | 53, 79 | 1,536 | 1,024 | 0.435 | 0.500 | 0.484 | 0.707 | 0.088 | 0.479 | 0.458 |
| 79 | 53, 67 | 1,536 | 1,024 | 0.389 | 0.500 | 0.519 | 0.785 | 0.117 | 0.666 | 0.611 |

Held-out-seed aggregate:

| Metric | Mean | Std |
|---|---:|---:|
| test records | 853.3 | 295.6 |
| test bad rate | 0.442 | 0.058 |
| token AUROC | 0.500 | 0.000 |
| hand score AUROC | 0.494 | 0.022 |
| calibrated AUROC | 0.742 | 0.040 |
| calibrated ECE | 0.100 | 0.015 |
| calibrated Pearson | 0.568 | 0.094 |
| calibrated Spearman | 0.532 | 0.076 |

Artifacts:

```text
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed53-syncer-current-6m/anchor_gradient_features_512.jsonl
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed67-syncer-current-6m/anchor_gradient_features_1024.jsonl
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed79-syncer-current-6m/anchor_gradient_features_1024.jsonl
experiment-results/EXP2/anchor_gradient_features_syncer_current_seeds53_67_79.jsonl
experiment-results/EXP2/calibrated_anchor_gradient_heldout_seed_53/
experiment-results/EXP2/calibrated_anchor_gradient_heldout_seed_67/
experiment-results/EXP2/calibrated_anchor_gradient_heldout_seed_79/
```

Interpretation:

- Current-state anchor-gradient features are much stronger than token count, freshness, historical alignment, and the previous hand score.
- The raw gradient-dot signal is stable across all three seeds.
- Held-out-seed calibration crosses the intended candidate-level signal threshold: mean AUROC is 0.742, and each held-out seed is above 0.70.
- Calibration error is near the target but not comfortably below it: mean ECE is 0.100, with seed 79 at 0.117.
- This passes the candidate scoring gate for the captured syncer-current records. It does not yet prove an end-to-end policy improves training. The next step is offline policy replay using anchor-gradient ranking and gating.

## Real HF T4 Diagnostic

This was a one-GPU AWS diagnostic, not the main result.

| Item | Value |
|---|---:|
| Region / instance | `us-west-2` / `g4dn.2xlarge` |
| GPU | 1x Tesla T4 |
| Logical learners | 12 |
| GPU slots | 1 |
| Token budget | 200,000 |
| Probe batches | 4 |
| Probe records | 166 |
| `round_token_cv_mean` | 0.000 |
| `round_token_cv_p95` | 0.000 |
| `negative_utility_rate` | 0.211 |
| `bad_strict_rate` | 0.096 |
| `token_auroc` | 0.500 |
| `freshness_auroc` | 0.832 |
| `alignment_auroc` | 0.597 |
| `hand_score_auroc` | 0.711 |
| `score_minus_token_auroc` | 0.211 |
| `utility_noise_estimate` | 0.00838 |

Artifact:

```text
experiment-results/EXP2/equal-token-late-smollm2-t4-debug/
experiment-results/EXP2/yeto-exp2-t4-debug-artifacts.tgz
```

Interpretation:

- The fixed-window/freeze-delay mechanics work in the real HF path.
- Token-count CV is exactly zero and token count is chance-level.
- Harmful fragments are visible, but the run is too small for the main gate.
- Freshness is the strongest signal in this tiny diagnostic, so it should not be treated as evidence for the current hand score.
- The baseline loss in this diagnostic was injected and should not be used for quality comparison.

## Real HF P4DE Seed-17 Run

This is the first larger equal-token run.

| Item | Value |
|---|---:|
| Region / instance | `us-east-1d` / `p4de.24xlarge` spot |
| GPU | 8x A100 80GB |
| Logical learners | 12 |
| GPU slots | 8 |
| Fragments | 12 |
| Quorum | 6 |
| Token budget | 6,200,000 |
| Steps per learner | 4,037 |
| Fixed window | 8,192 tokens / 64 microsteps |
| Probe batches | 8 |
| Wall time for async arm | 1,844 s |
| Probe records | 8,668 |

Final probe summary:

| Metric | Value |
|---|---:|
| `round_token_cv_mean` | 0.000 |
| `round_token_cv_p95` | 0.000 |
| `negative_utility_rate` | 0.400 |
| `bad_strict_rate` | 0.140 |
| `utility_noise_estimate` | 0.00262 |
| utility p05 | -0.00385 |
| utility p50 | 0.00078 |
| utility p95 | 0.00769 |
| `token_auroc` | 0.500 |
| `freshness_auroc` | 0.515 |
| `alignment_auroc` | 0.505 |
| `hand_score_auroc` | 0.495 |
| `score_minus_token_auroc` | -0.0047 |

Incidental eval output:

| Arm | Eval loss |
|---|---:|
| base, untrained | 2.3180 |
| sync baseline, injected | 2.2161 |
| m12 | 2.0559 |

The injected sync baseline is a placeholder from an earlier setting. It should not be used to compare quality in this p4de run. The useful result here is the fragment probe summary.

Artifact:

```text
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed17-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed17-6m.tgz
```

Interpretation:

- Measurement cleanliness passes: token-count CV is exactly zero.
- The equal-token phenomenon is present: utility has a real spread, `negative_utility_rate` is about 40%, and strict bad fragments are about 14%.
- Token count fails as a utility proxy under the controlled protocol: token AUROC is 0.500 with zero correlation by construction.
- The current hand score fails: `hand_score_auroc` is 0.495, and all simple signals are near chance in this seed.
- This is one seed and still uses the learner-known oracle. It is a strong measurement result, not a final scoring result.
- The syncer-current replay path was added after this run and is evaluated in the later replicated diagnostics. Stale-replay depth mode remains pending.

## Real HF P4DE Three-Seed Replication

These runs use the same real HF model/data path and fixed-window freeze-delay protocol as seed 17.

Common setting:

| Item | Value |
|---|---:|
| Region / instance | `us-east-1d` / `p4de.24xlarge` spot |
| GPU | 8x A100 80GB |
| Model | `HuggingFaceTB/SmolLM2-135M` |
| Dataset | `trl-lib/Capybara` |
| Logical learners | 12 |
| GPU slots | 8 |
| Fragments | 12 |
| Quorum | 6 |
| Token budget | 6,200,000 |
| Steps per learner | 4,037 |
| Fixed window | 8,192 tokens / 64 microsteps |
| Probe batches | 8 |

Per-seed probe summary:

| Seed | Records | Token CV mean | Token CV p95 | Negative utility | Bad strict | Token AUROC | Freshness AUROC | Alignment AUROC | Hand score AUROC | Score - token | Utility p05 | Utility p50 | Utility p95 | Noise est. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 8,668 | 0.000 | 0.000 | 0.400 | 0.140 | 0.500 | 0.515 | 0.505 | 0.495 | -0.0047 | -0.00385 | 0.00078 | 0.00769 | 0.00262 |
| 23 | 8,626 | 0.000 | 0.000 | 0.404 | 0.141 | 0.500 | 0.528 | 0.485 | 0.520 | 0.0195 | -0.00426 | 0.00077 | 0.00720 | 0.00277 |
| 41 | 8,725 | 0.000 | 0.000 | 0.386 | 0.133 | 0.500 | 0.528 | 0.489 | 0.496 | -0.0042 | -0.00447 | 0.00100 | 0.00870 | 0.00302 |

Aggregate:

| Metric | Mean | Std |
|---|---:|---:|
| Records | 8,673.0 | 49.7 |
| `round_token_cv_mean` | 0.000 | 0.000 |
| `round_token_cv_p95` | 0.000 | 0.000 |
| `negative_utility_rate` | 0.397 | 0.009 |
| `bad_strict_rate` | 0.138 | 0.004 |
| `token_auroc` | 0.500 | 0.000 |
| `freshness_auroc` | 0.524 | 0.007 |
| `alignment_auroc` | 0.493 | 0.010 |
| `hand_score_auroc` | 0.504 | 0.014 |
| `score_minus_token_auroc` | 0.0036 | 0.0138 |
| `utility_noise_estimate` | 0.00280 | 0.00021 |
| utility p05 | -0.00419 | 0.00031 |
| utility p50 | 0.00085 | 0.00013 |
| utility p95 | 0.00786 | 0.00077 |

Artifacts:

```text
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed17-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed17-6m.tgz
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed23-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed23-6m.tgz
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed41-6m/
experiment-results/EXP2/equal-token-late-smollm2-p4de-seed41-6m.tgz
```

Interpretation:

- Measurement cleanliness replicates: all three p4de seeds have exactly zero token-count CV.
- Equal-token utility spread replicates: negative-utility fragments remain common, with a mean rate of 39.7%; the stricter `utility + utility_se < 0` rate averages 13.8%.
- Token count is controlled away and provides no separation: token AUROC is exactly 0.500 in all three seeds.
- Utility spread is larger than probe noise: mean p95-p05 is about 0.0121, compared with mean `utility_noise_estimate` of 0.0028.
- The current hand score fails the signal gate: mean AUROC is 0.504, and the best seed reaches only 0.520.
- Freshness is weakly above chance in seeds 23 and 41, but not enough to be a useful standalone signal.
- These results justify continuing measurement work, but the scoring side must be redesigned before an end-to-end policy run.

## Local Smoke Check

A tiny local fixture was used only to verify wiring:

| Item | Value |
|---|---:|
| Model | `/tmp/yeto-exp1-real/tiny-llama` |
| Dataset | `/tmp/yeto-exp1-real/chat.jsonl` |
| Learners | 2 |
| Token budget | 2,048 |
| Probe records | 44 |
| `round_token_cv_mean` | 0.000 |
| `round_token_cv_p95` | 0.000 |
| `negative_utility_rate` | 0.068 |
| `bad_strict_rate` | 0.068 |
| `token_auroc` | 0.500 |
| `hand_score_auroc` | 0.927 |

This smoke result is not evidence for the main question. It only verifies that fixed-window snapshots, frozen payload packing, delayed push, probe logging, and summary metrics run end to end.
