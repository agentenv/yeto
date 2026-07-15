# LM DiLoCo Benchmark

## Question

The benchmark tests one product claim:

> At the same model, data, LoRA recipe, device count, per-rank batch, and
> raw-token budget, splitting the devices into independent DiLoCo islands
> should produce held-out causal-LM loss close to one synchronous process
> group.

It is a LoRA quality and systems-ablation benchmark. It is not a model
leaderboard, an instruction-following evaluation, a cloud provisioning test,
or a WAN throughput emulator.

The executable harness is `scripts/compare_diloco.py`.

The synchronization arms are shared with diffusion because they exercise the
same syncer algorithms. The workload contract, budget, data accounting, quality
metric, and diagnostics are LM-specific; diffusion sample/noise/shape controls
are not part of this benchmark.

## Fairness Contract

For an arm with `M` islands and `G` ranks per island:

- `baseline-mM` uses one synchronous process group with `M * G` ranks;
- the DiLoCo arm uses `M` independent process groups with `G` ranks each;
- both use the same per-rank micro batch and gradient accumulation;
- both run the same number of optimizer steps;
- both therefore process the same number of raw packed tokens;
- corresponding ranks consume the same conversation shards in the same order;
- the harness verifies that both sides process exactly the same number of
  positive-weight training target tokens;
- model id, train/eval rows, sequence length, LoRA configuration, optimizer,
  learning-rate schedule, loss mask, and root training seed are identical;
- each DiLoCo arm is compared only with the baseline having the same `M`.

The step count is:

```text
ceil(token_budget /
     (micro_batch * grad_accum * seq_len * M * G))
```

The ceiling can overshoot the requested token budget by less than one global
parallel step. The report records the actual processed count.

This corrects the old comparison in which a one-island baseline used fewer
devices and a different global batch. That older setup was useful for quality
probes, but its wall time, throughput, learning-rate schedule, and optimizer
step count were not directly comparable with an `M`-island arm.

The training budget counts all packed input tokens because prompt/context
tokens still determine attention compute and synchronization cadence. The
benchmark separately records target tokens and target density. With the
default `--train-on assistant`, user and system context consumes compute but
does not enter the optimization or CE denominator. A target-token mismatch
between an arm and its baseline invalidates the run instead of being hidden by
an equal raw-token count.

## Authoritative Artifact

A DiLoCo learner applies delayed global broadcasts while continuing local
optimization. Its local adapter is therefore not the global result.

The benchmark checkpoints the real Rust syncer, rebuilds the exact trainable
layout with `yeto.export`, and evaluates the exported merged adapter. It never
scores a learner's local output.

The synchronous baseline is evaluated from the adapter saved by rank 0 after
the final synchronized optimizer step.

## Data And Evaluation

The final `--eval-rows` rows are removed before training. Every run trains on
the same preceding rows and evaluates on the same held-out rows. The benchmark
materializes `messages` and optional `tools` fields into fixed local JSONL
files so a changing remote dataset cannot affect arms within a run.

Evaluation packs the held-out conversations with the same tokenizer, sequence
length, and `--train-on` mask as training. Its primary metric is:

```text
sum(masked causal cross entropy) / number of positive-weight target tokens
```

The report also shows perplexity (`exp(CE/token)`) and the exact held-out target
token count. Every artifact must produce the same count as the base model.

Unlike diffusion flow-matching evaluation, LM CE in `model.eval()` has no
sampled timestep or noise. One evaluation pass is sufficient for an artifact.
Optimization still varies with LoRA initialization, data order, and dropout,
so the benchmark runs three training seeds by default and reports mean and
standard deviation.

## Recommended Dataset

The recommended compact LM benchmark dataset is:

```text
HuggingFaceH4/Multilingual-Thinking
```

Selection reasons:

- released in August 2025, newer than the historical Lean-Workbook setup;
- Apache-2.0 license;
- exactly 1,000 train rows;
- about 5.3 MB compressed and 8.9 MB materialized;
- already exposes standard `messages` rows with system, user, and assistant
  roles, so no schema conversion is required;
- multilingual and instruction-diverse rather than restricted to one math or
  code domain;
- large enough for the 500k-token benchmark without making dataset download or
  held-out evaluation dominate the run.

The pinned dataset revision used when this recommendation was made is:

```text
f423949d2726f5a5633ea10ac45bc1ea1e0de6e7
```

Rows also contain a separate assistant `thinking` field. Yeto intentionally
trains and evaluates the normal assistant `content` field only; hidden
reasoning is not folded into the target. This keeps the benchmark a standard
instruction-SFT comparison rather than a reasoning-trace distillation test.

Recommended first formal profile:

```text
token budget: 500,000 raw tokens
eval rows:    64
sequence:     512
train mask:   assistant
seeds:        17,29,43
```

For a durable formal result, materialize the pinned revision to a versioned S3
prefix before running. A mutable Hugging Face dataset id is convenient for a
smoke run but is not a permanent experiment input.

## LM Workload Profiles

Sequence length and assistant-target density materially change both LM compute
and the number of optimization targets per raw token. They are workload
profiles, not DiLoCo algorithm arms, and must not be mixed inside one paired
comparison.

Use the historical `512`-token profile when reproducing the existing
gemma4/Lean result. For a production decision, run a separate benchmark with
the actual production sequence length, chat template, `--train-on` mode, LoRA
targets, and batch recipe. Compare arms only within a profile. This is more
meaningful than adding a generic diffusion-style "large sample" arm.

## Arms

| arm | change from production M=2 | question |
|---|---|---|
| `m2` | none | Does the production-shaped two-island path match sync? |
| `m4` | four islands | How does quality scale with more independent islands? |
| `alpha0` | overwrite broadcasts | Is local/global blending helping? |
| `q4` | q4 learner pushes | What quality cost buys lower learner egress? |
| `serial` | one fragment round in flight | Does pipelining add harmful staleness? |
| `noheloco` | no delta correction | Is HeLoCo correcting useful stale directions? |
| `strided` | depth-interleaved fragments | Does fragment construction affect quality? |
| `iso` | Iso-C matrix aggregation | Does isotropic matrix merging improve the LM path? |
| `direct-rda` | no Nesterov, full outer step, overwrite | Is outer-optimizer gain causing a gap? |
| `unthrottled` | no H target | How sensitive is LM training to low-latency over-syncing? |

`m2` and `m4` use the production `H=24` target. `unthrottled` is the explicit
localhost/LAN stress arm.

`direct-rda` is not plain parameter averaging. Non-embedding fragments still
use radial-directional averaging. The arm removes Nesterov momentum, changes
the outer learning rate to 1, and overwrites local parameters so the merged RDA
delta is applied directly.

The previous names map as follows:

| old name | current name |
|---|---|
| `m2h24` | `m2` |
| `m2` with H disabled | `unthrottled` |
| `avg` | `direct-rda` |

## Systems Measurements

Held-out CE answers the quality question. The event tape and run metadata show
whether the intended synchronization actually occurred:

- measured mean inner steps per fragment response (`H`);
- mean raw tokens per fragment response, the LM-relevant sync interval;
- total target tokens, target density, and target-token throughput;
- responder participation rate;
- mean fragment-version staleness;
- merge and sync duration counts;
- estimated accepted tensor bytes for initialization, pushes, and broadcasts;
- wall time, raw tokens/second, GPU-hours, and optional estimated cost.

These are diagnostics, not replacements for held-out quality. Localhost wall
time measures the algorithms on one host; it does not predict cross-region TCP
throughput. Zero-worker tokenization favors reproducible pairing over maximum
input-pipeline throughput, so reported tokens/second is an arm comparison, not
the learner's absolute production throughput ceiling.

## Reproducibility

The harness uses explicit batch sizes and seeded streaming tokenization with
DataLoader workers fixed at zero. This avoids making every distributed rank
pre-tokenize a duplicate copy of the dataset while keeping row sharding and
order reproducible. A root seed controls LoRA initialization. After
initialization, each learner/rank receives a deterministic derived seed for
data order and training randomness. This keeps every island's starting adapter
identical while making repeated runs reproducible. With the zero-worker
stream, a DiLoCo learner/rank and its corresponding rank in `baseline-mM` use
the same derived seed and the same row shard.

CUDA kernels are not forced into bitwise-deterministic implementations.
Repeated seeds estimate training variance rather than promising identical
floating-point results on every machine.

## Running

`--data` accepts a local path, a Hugging Face dataset id, or an S3 prefix. S3
prefixes are downloaded read-only with `aws s3 sync` into
`<work-dir>/source-data`. The AWS CLI and ambient read credentials must be
available. `--dry-run` validates topology and budgets without loading data or
models.

Dry-run a plan whose largest arm uses eight GPUs:

```bash
python scripts/compare_diloco.py \
  --model gemma4 \
  --data HuggingFaceH4/Multilingual-Thinking \
  --settings m2,m4,unthrottled \
  --learner-gpus 2 \
  --device cuda --eval-device cuda --shard fsdp \
  --token-budget 500000 \
  --dry-run
```

Run the benchmark:

```bash
python scripts/compare_diloco.py \
  --model gemma4 \
  --data HuggingFaceH4/Multilingual-Thinking \
  --settings all \
  --learner-gpus 2 \
  --device cuda --eval-device cuda --shard fsdp \
  --token-budget 500000 \
  --seeds 17,29,43 \
  --overwrite
```

Outputs:

- `config.json`: arguments, arm definitions, and the fairness contract;
- `results.jsonl`: one record per seed and arm, written after every run;
- `summary.json`: cross-seed aggregates and paired deltas;
- `report.md`: the human-readable comparison table;
- learner, syncer, export, and evaluation logs under `--work-dir`.

Use `--resume` after interruption. The harness verifies the original arguments,
arm definitions, implementation fingerprint, serialized split hashes, and local
source-data hash when applicable before keeping completed `(kind, arm, seed)`
records. It reuses the original train/eval splits and executes only missing
runs. Legacy runs without an immutable resume manifest must restart with
`--overwrite`.

## Interpretation Rules

- Compare a DiLoCo arm only with `baseline-mM` at the same `M` and seed.
- Do not conclude from one training seed.
- Confirm that the synchronous baseline improves over the untrained base.
- Treat any train or held-out target-token mismatch as an invalid run. Equal
  raw-token counts alone are insufficient for assistant-masked SFT.
- Treat unexpected measured `H`, low participation, or no completed merges as
  an invalid run rather than an algorithm result.
- Compare throughput only within the same `M`, because `m2` and `m4` may have
  different ceiling overshoot and total device counts.
- Treat q4 byte estimates as payload accounting, not a WAN transfer benchmark.
- Use task accuracy, generation quality, or judge-based evaluation only as a
  separate evaluator over saved artifacts; do not change the training path.

## LM And Diffusion Project Comparison

The synchronization ablation matrix shares nine presets across both domains.
Each benchmark has the untrained base and matching synchronous baselines. The
LM harness additionally includes the `iso` matrix-aggregation arm.

| project | LM benchmark | Diffusion benchmark |
|---|---|---|
| untrained reference | `base` | `base` |
| equal-hardware sync | `baseline-mM` | `baseline-mM` |
| production topology | `m2`, `m4` | `m2`, `m4` |
| merge/blending | `alpha0`, `direct-rda` | `alpha0`, `direct-rda` |
| transport/scheduler | `q4`, `serial`, `unthrottled` | `q4`, `serial`, `unthrottled` |
| correction/layout | `noheloco`, `strided` | `noheloco`, `strided` |
| matrix aggregation | `iso` | not currently included |

The domain-specific parts differ:

| dimension | LM | Diffusion |
|---|---|---|
| budget unit | raw packed tokens plus verified target tokens | training samples |
| input controls | sequence length, assistant/all mask | height, width, resize, frames, FPS, caching |
| primary metric | masked CE/token and perplexity | flow-matching loss per predicted element |
| training randomness | paired logical-rank streams | paired logical-rank streams |
| eval randomness | deterministic forward pass | paired timestep/noise draws with repeats |
| dataset row | conversation plus optional tools | prompt plus image/video/latent fields |
| saved adapter | causal-LM LoRA | diffusion-component LoRA |
| sync interval diagnostic | raw tokens per response | samples/steps per response |

The LM update adopts the diffusion benchmark's equal-hardware baselines,
production-shaped preset names, repeated seeds, S3 staging, resume behavior,
syncer-tape diagnostics, traffic estimates, and structured reports. The
Both harnesses pair logical-rank training streams. The LM harness additionally
checks exact target-token counts and reports target density, perplexity, and
tokens-per-sync diagnostics. The
diffusion-only shape, frame, noise-repeat, and sample-budget controls are
deliberately not copied because they do not apply to causal language modeling.

## Current Scope

The harness covers LoRA causal-LM fine-tuning with masked cross entropy. Full
parameter synchronization, multi-node launch orchestration, WAN fault
injection, task-suite evaluation, and checkpoint-time learning curves are
separate experiments.
