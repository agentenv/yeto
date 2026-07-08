# EXP1: Balanced Data, Bad Fragments

Date: 2026-07-07

## Question

Does balanced preprocessing make token count a reliable proxy for asynchronous fragment utility?

Short answer: the controlled proxy says no, but the first real HF run is not clean enough to support that answer. With equal data distribution across learners and near-equal token counts, the proxy produced widely different one-step utility and token count was effectively chance as a predictor. In the real learner/syncer run, token counters varied enough that token count became predictive, so the next run must isolate token count more carefully.

Follow-up status, 2026-07-08: EXP2 now isolates token count with fixed-token fragment snapshots and post-snapshot delay on a real Hugging Face model/data path. The syncer-current three-seed p4de replication keeps token CV at zero and token AUROC at 0.500, while negative-utility fragments average 44.2%. The original hand score still fails, but sampled merge replay shows actual merge headroom and an anchor-gradient score passes the held-out-seed candidate scoring diagnostic. Full details are in `docs/EXP2.md`.

## Scope

This is a controlled local probe, not a full language-model training run.

I first used a balanced stochastic quadratic objective with cross-fragment coupling so that every learner samples the same underlying data distribution. The only stressors were asynchronous response age, heterogeneous response groups, local optimizer noise, and rare optimizer-state bursts. This isolated the measurement question before spending cloud GPU time.

A later section reports a short GPU-backed run on AWS with the real learner/syncer path and the new fragment utility instrumentation.

## Reproduction

Script added:

```bash
scripts/fragment_utility_probe.py
```

Reported single-seed command:

```bash
python scripts/fragment_utility_probe.py --out-dir /tmp/yeto-exp1-final
```

Seed sweep:

```bash
for seed in 11 12 13 14 15; do
  python scripts/fragment_utility_probe.py \
    --seed "$seed" \
    --out-dir "/tmp/yeto-exp1-seed-$seed"
done
```

Environment note: the script depends on `numpy`.

## Setup

| Parameter | Value |
|---|---:|
| Learners | 12 |
| Fragments | 12 |
| Dimension | 768 |
| Rounds | 600 |
| Candidates per seed | 7,200 |
| Local steps per response | 36 |
| Sequence length used for token accounting | 512 |
| Token jitter | 3.5% |
| Response groups | fast / medium / delayed |
| Objective distribution | identical for all learners |

Oracle utility for learner `i`, fragment `f`, round `t`:

```text
u(i,f,t) = L(theta_t) - L(theta_t with fragment f updated by eta * delta_i,f,t)
```

Positive utility means the returned fragment improves the current probe objective for that one-step application. Negative utility means it is locally harmful at the current global state.

## Single-Seed Result

Seed: 11

| Metric | Value |
|---|---:|
| Candidate fragments | 7,200 |
| Negative-utility fragment rate | 71.25% |
| Mean bad token-weighted mass per round | 71.25% |
| Rounds where merged token-weighted update was negative utility | 33.17% |
| Mean token-count CV per round | 3.28% |
| p95 token-count CV per round | 4.49% |

Utility quantiles:

| Quantile | Utility |
|---|---:|
| p05 | -0.003734 |
| p50 | -0.000120 |
| p95 | 0.110670 |

The token counts were tightly balanced, but the returned fragments were not. The token-weighted merge included substantial harmful mass even though the learners had the same data distribution.

## Signal Quality

Seed 11:

| Signal | Pearson with utility | Spearman with utility | Bad-fragment AUROC | Calibration error |
|---|---:|---:|---:|---:|
| token count | -0.001 | -0.002 | 0.503 | 0.250 |
| freshness | 0.027 | 0.039 | 0.507 | 0.244 |
| alignment | 0.147 | -0.105 | 0.589 | 0.287 |
| norm anomaly | -0.039 | -0.020 | 0.445 | 0.272 |
| combined score | 0.189 | 0.185 | 0.715 | 0.071 |

Five-seed mean over seeds 11-15:

| Metric | Mean | Std |
|---|---:|---:|
| Negative-utility fragment rate | 70.52% | 1.54% |
| Mean bad token-weighted mass | 70.53% | 1.54% |
| Negative merged-update round rate | 32.67% | 2.12% |
| Mean token-count CV | 3.29% | 0.02% |
| token-count AUROC | 0.497 | 0.008 |
| freshness AUROC | 0.510 | 0.002 |
| alignment AUROC | 0.601 | 0.016 |
| combined-score AUROC | 0.733 | 0.018 |
| combined-score calibration error | 0.072 | 0.013 |

## Interpretation

The phenomenon check passes:

- Token count was balanced but did not predict utility.
- Harmful fragments appeared frequently under balanced data.
- Token-weighted merging included a large amount of harmful fragment mass.
- Freshness alone was not enough.
- Alignment helped, but the combined score ranked harmful fragments better and was much better calibrated in this stress setting.

The method claim is not fully validated yet:

- This probe is a controlled proxy, not a transformer pretraining run.
- The negative-fragment rate is intentionally high because the stress setting includes delayed responses and rare optimizer-state bursts.
- The real learner path now has utility instrumentation, but the actual stressed language-model run has not been executed yet.

## Decision

Proceed to a GPU-backed stress run using the real utility instrumentation.

Required next run step:

1. Use the learner `--probe-data` path to log utility records.
2. Run at least 12 logical learners with balanced data.
3. Add explicit delay / speed heterogeneity / late-fragment stress.
4. Summarize the resulting probe JSONL with `scripts/summarize_fragment_probe.py`.

Go condition for the next run:

- token-count AUROC remains near chance,
- negative-utility fragments are visible under balanced data,
- combined score beats token count, freshness, and alignment on ranking or calibration,
- the instrumentation overhead is low enough to keep the learner loop usable.

## Real-Harness Instrumentation

Implemented opt-in fragment utility logging in the learner.

New learner flags:

```bash
--probe-data <path-or-dataset>
--probe-log <jsonl-path>
--probe-every <N>
--probe-batches <N>
--probe-batch-size <N>
--probe-max-rows <N>
--probe-outer-lr <float>
--probe-freshness-scale <float>
```

The probe runs at the pull-response boundary. For each sampled candidate it logs:

- learner id,
- fragment id,
- pull step,
- base version,
- local step,
- local steps since reset,
- token count since reset,
- fragment age,
- freshness,
- alignment,
- uncertainty,
- norm anomaly,
- combined score,
- update norm,
- base probe loss,
- trial probe loss,
- utility.

The utility scope is `learner_known_global`: the learner temporarily resets trainable fragments to the last broadcast anchors it has applied, applies the candidate fragment delta, evaluates a fixed probe batch, then restores local training parameters. This is the practical low-overhead probe available inside the learner process. An exact syncer-state oracle still requires either a Python sidecar with access to the syncer state or offline replay of captured pushes.

Comparison harness pass-through was added:

```bash
python scripts/compare_diloco.py \
  --model lfm25-230m \
  --data <chat.jsonl> \
  --settings m2 \
  --token-budget 50000 \
  --seq-len 256 \
  --micro-batch-size 1 \
  --device cuda \
  --probe-data eval \
  --probe-every 4 \
  --probe-batches 2 \
  --probe-batch-size 1
```

Probe summary:

```bash
python scripts/summarize_fragment_probe.py \
  compare-work/m2/fragment_probe_learner_*.jsonl \
  --out compare-report/fragment_probe_summary.json
```

Verification completed:

- `python3 -m py_compile yeto/learner.py scripts/compare_diloco.py scripts/fragment_utility_probe.py scripts/summarize_fragment_probe.py`
- synthetic JSONL summary smoke test for `scripts/summarize_fragment_probe.py`
- `scripts/compare_diloco.py --dry-run` with `--probe-data eval`

## Real-Harness Smoke Run

I created a disposable local Python runtime under `/tmp/yeto-runtime`, generated a tiny local Llama-style causal LM checkpoint and a small balanced chat JSONL under `/tmp/yeto-exp1-real`, then ran the real learner/syncer path on CPU.

Command:

```bash
/tmp/yeto-runtime/bin/python scripts/compare_diloco.py \
  --model /tmp/yeto-exp1-real/tiny-llama \
  --data /tmp/yeto-exp1-real/chat.jsonl \
  --token-budget 2048 \
  --settings m2 \
  --seq-len 32 \
  --micro-batch-size 1 \
  --eval-rows 8 \
  --max-rows 64 \
  --device cpu \
  --probe-data eval \
  --probe-every 1 \
  --probe-batches 1 \
  --probe-batch-size 1 \
  --probe-max-rows 8 \
  --arm-timeout-min 10 \
  --work-dir /tmp/yeto-exp1-real/work \
  --report-dir /tmp/yeto-exp1-real/report
```

Training result:

| Arm | Eval loss/token | Wall time | Delta vs sync baseline |
|---|---:|---:|---:|
| base | 3.5000 | 0s | n/a |
| sync baseline | 3.1976 | 4s | n/a |
| async m2 | 3.2118 | 4s | +0.44% |

Probe records:

| Log | Records |
|---|---:|
| `/tmp/yeto-exp1-real/work/m2/fragment_probe_learner_0.jsonl` | 62 |
| `/tmp/yeto-exp1-real/work/m2/fragment_probe_learner_1.jsonl` | 2 |
| total | 64 |

Probe summary:

| Metric | Value |
|---|---:|
| Negative-utility fragment rate | 0.00% |
| Utility p05 | 0.000060 |
| Utility p50 | 0.000656 |
| Utility p95 | 0.006547 |
| token-count Pearson with utility | -0.003 |
| alignment Pearson with utility | 0.332 |
| combined-score Pearson with utility | -0.000 |

Interpretation:

- The instrumentation executes end to end in the real learner/syncer path.
- The JSONL records contain the fields needed for the utility-vs-signal tables.
- This tiny CPU smoke run is not expected to reproduce harmful fragments: it has only two learners, a toy model, a tiny token budget, localhost networking, and no explicit delay/failure stress.
- AWS was not used for this smoke run because a local disposable runtime was sufficient to validate the new logging path.

## GPU Stress Run

Date: 2026-07-08 UTC / 2026-07-07 Pacific

Cloud setup:

| Item | Value |
|---|---|
| Region | `us-west-2` |
| Instance | `g4dn.2xlarge` |
| GPU | 1 x Tesla T4, 15 GiB |
| Market | on-demand |
| Reason for on-demand | spot capacity was unavailable for the tested `g5.12xlarge`, `g6.4xlarge`, and `g4dn.2xlarge` requests |
| Cleanup | instance terminated; temporary key pair and security group deleted |

Experiment setup:

| Parameter | Value |
|---|---:|
| Logical learners | 12 |
| Quorum | 6 |
| Fragments | 12 |
| Token budget per arm | 24,576 |
| Sequence length | 64 |
| Micro batch size | 1 |
| Inner LR | 0.005 |
| LoRA rank / alpha | 4 / 8 |
| Probe batches | 2 |
| Probe records | every answered candidate |

The model was a small generated local Llama-style checkpoint (`hidden_size=128`, 4 layers) with a balanced synthetic chat JSONL. This was chosen so 12 logical learners could run concurrently on one T4. It is a real learner/syncer GPU run, but it is still a small stress test, not a scale result.

Artifacts:

```text
experiment-results/EXP1-gpu/step-and-push-stress/
experiment-results/EXP1-gpu/late-push-stress/
```

Each artifact directory contains:

- `report.md`
- `results.jsonl`
- `run.log`
- `fragment_probe_summary.json`
- `fragment_probe_diagnostics.json`
- `probe-jsonl.tar.gz`

### Variant A: Step + Late-Push Stress

Stress profile:

- per-learner optimizer-step sleeps: `0,10,60,0,20,80,0,30,100,0,50,120` ms
- per-learner push delays: `0,0,80,0,20,120,0,40,160,0,60,200` ms
- jitter: up to 10 ms

Training result:

| Arm | Eval loss/token | Wall time | Delta vs sync baseline |
|---|---:|---:|---:|
| base | 5.6289 | 0s | n/a |
| sync baseline | 4.3816 | 18s | n/a |
| async m12 | 4.6290 | 38s | +5.65% |

Probe result:

| Metric | Value |
|---|---:|
| Probe records | 254 |
| Negative-utility fragment rate | 36.61% |
| Mean bad token-weighted mass | 31.33% |
| Utility p05 | -0.042073 |
| Utility p50 | 0.003168 |
| Utility p95 | 0.056948 |

Signal table:

| Signal | Pearson with utility | Spearman with utility | Bad-fragment AUROC | Calibration error |
|---|---:|---:|---:|---:|
| token count | 0.251 | 0.293 | 0.598 | 0.182 |
| freshness | -0.187 | -0.056 | 0.528 | 0.254 |
| alignment | -0.021 | 0.074 | 0.427 | 0.309 |
| norm anomaly | -0.009 | 0.041 | 0.575 | 0.137 |
| combined score | -0.119 | 0.038 | 0.515 | 0.484 |

Interpretation:

- This variant produced many harmful fragments.
- It is not the clean token-count test because step-speed delays changed local step and token counters. Token count became a weak predictor here.
- Use this variant as a stress-path validation, not as the main token-count failure evidence.

### Variant B: Late-Push Stress Only

Stress profile:

- no optimizer-step sleep
- per-learner push delays: `0,50,100,150,200,250,0,75,150,225,300,375` ms
- jitter: up to 20 ms

Training result:

| Arm | Eval loss/token | Wall time | Delta vs sync baseline |
|---|---:|---:|---:|
| base | 5.6289 | 0s | n/a |
| sync baseline | 4.3816 | injected from Variant A | n/a |
| async m12 | 4.6512 | 44s | +6.15% |

Probe result:

| Metric | Value |
|---|---:|
| Probe records | 201 |
| Negative-utility fragment rate | 28.86% |
| Mean bad token-weighted mass | 26.84% |
| Utility p05 | -0.018199 |
| Utility p50 | 0.003134 |
| Utility p95 | 0.053287 |

Signal table:

| Signal | Pearson with utility | Spearman with utility | Bad-fragment AUROC | Calibration error |
|---|---:|---:|---:|---:|
| token count | 0.105 | 0.088 | 0.498 | 0.267 |
| freshness | 0.058 | -0.007 | 0.505 | 0.284 |
| alignment | 0.149 | 0.162 | 0.498 | 0.280 |
| norm anomaly | -0.014 | -0.012 | 0.516 | 0.187 |
| combined score | 0.246 | 0.258 | 0.578 | 0.463 |

Interpretation:

- This is the cleaner result for the current run.
- Harmful fragments appeared under balanced data and late-fragment stress.
- Token count was at chance for bad-fragment detection: AUROC `0.498`.
- Freshness and alignment alone were also near chance in AUROC.
- The combined score improved ranking over token count in this small run, but calibration was poor and the margin is modest.
- Token counts still varied because queued pulls and quorum timing changed counters; the important observation here is that token count variation did not rank harmful fragments.

## GPU Run Decision

This GPU stress run clears the first practical gate:

- real learner/syncer instrumentation works on GPU,
- 12 logical learners can log fragment utility records,
- negative-utility fragments are visible,
- the late-only variant shows token-count AUROC near chance.

It does not yet clear the stronger result gate:

- the model is intentionally small,
- the run uses one T4 rather than multi-GPU islands,
- the combined score is only modestly better than token count,
- token counters were not as controlled as the ideal balanced-token setup.

Next run should use a larger small model and more physical GPUs, with a stress mode that delays responses without changing local token counters as much.

## Real Hugging Face Dataset Run

Date: 2026-07-08 UTC / 2026-07-07 Pacific

This run replaced the generated dataset and generated model with public Hugging Face assets:

| Item | Value |
|---|---|
| Dataset | `trl-lib/Capybara` |
| Dataset schema | `messages`, `source`, `num_turns` |
| Model | `HuggingFaceTB/SmolLM2-135M` |
| Instance | `g4dn.2xlarge` |
| GPU | 1 x Tesla T4, 15 GiB |
| Market | on-demand |
| Cleanup | instance terminated; temporary key pair and security group deleted |

The dataset already has a `messages` column, so it used the normal Yeto data loader. I added deterministic row shuffling in the comparison harness and ran with `--shuffle-rows-seed 17` to reduce row-order sharding artifacts.

Command shape:

```bash
python scripts/compare_diloco.py \
  --model HuggingFaceTB/SmolLM2-135M \
  --data trl-lib/Capybara \
  --token-budget 24576 \
  --settings m12 \
  --seq-len 128 \
  --micro-batch-size 1 \
  --inner-lr 0.001 \
  --lora-r 2 \
  --lora-alpha 4 \
  --eval-rows 64 \
  --max-rows 1200 \
  --shuffle-rows-seed 17 \
  --device cuda \
  --gpu-slots 1 \
  --probe-data eval \
  --probe-every 1 \
  --probe-batches 2 \
  --probe-batch-size 1 \
  --probe-max-rows 64 \
  --probe-freshness-scale 12 \
  --learner-step-sleep-ms 0 \
  --learner-push-delay-ms 0,50,100,150,200,250,0,75,150,225,300,375 \
  --learner-delay-jitter-ms 20
```

Artifacts:

```text
experiment-results/EXP1-gpu/capybara-smollm2-late-push/
```

Training result:

| Arm | Eval loss/token | Wall time | Delta vs sync baseline |
|---|---:|---:|---:|
| base | 2.4207 | 0s | n/a |
| sync baseline | 2.2161 | 54s | n/a |
| async m12 | 2.3798 | 86s | +7.39% |

Probe result:

| Metric | Value |
|---|---:|
| Probe records | 265 |
| Negative-utility fragment rate | 54.34% |
| Mean bad token-weighted mass | 46.01% |
| Utility p05 | -0.008431 |
| Utility p50 | -0.000789 |
| Utility p95 | 0.008470 |
| c_tokens CV | 45.74% |
| round token CV mean | 21.85% |
| round token CV p95 | 34.28% |

Signal table:

| Signal | Pearson with utility | Spearman with utility | Bad-fragment AUROC | Calibration error |
|---|---:|---:|---:|---:|
| token count | 0.258 | 0.255 | 0.622 | 0.136 |
| freshness | -0.232 | -0.223 | 0.408 | 0.323 |
| alignment | 0.173 | 0.213 | 0.586 | 0.183 |
| norm anomaly | -0.055 | -0.072 | 0.482 | 0.245 |
| combined score | -0.092 | -0.131 | 0.444 | 0.349 |

Interpretation:

- The real HF dataset/model run produced many negative-utility fragments.
- This run does not support the token-count failure claim: token count had bad-fragment AUROC `0.622`.
- The combined score performed worse than token count on this run.
- Token counters still varied substantially under late-push stress, even with no step sleep, because quorum/pipeline timing changes how long fragments go between resets.
- This is a useful negative result: real data/model plus the current stress protocol does not yet reproduce the clean proxy result.

Next adjustment:

- keep the real HF dataset/model,
- reduce token-counter variation by changing the stress mechanism or quorum schedule,
- log per-learner dataset/source mixture diagnostics,
- rerun with more physical GPUs so logical learner contention on one T4 is not a hidden stressor.

## EXP1 Conclusion

EXP1 established three useful facts:

- The controlled proxy shows that token-balanced asynchronous fragments can have widely varying one-step utility.
- The real learner/syncer instrumentation works and produces analyzable fragment utility records.
- GPU stress runs can produce negative-utility fragments.

EXP1 did not establish the stronger real-LM claim. In the real Hugging Face run, the current late-push stress protocol did not isolate token count: token counters varied substantially, and token count became predictive of bad fragments. That means the current stress setup mixes token exposure, queue timing, reset timing, quorum timing, learner progress, and fragment age.

The next experiment must enforce equal-token local response windows and apply artificial delay after the fragment delta snapshot is generated. The goal is to make token-count variation small enough that utility spread, if present, cannot be explained by token-count differences.

Required measurement thresholds for the next run:

- round token CV mean below 3%,
- round token CV p95 below 5%,
- at least 5,000 probe records for main analysis,
- multiple seeds for the real result,
- negative-utility fragment rate above 10%,
- token-count bad-fragment AUROC near chance.
