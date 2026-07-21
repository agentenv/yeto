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
length, `--train-on` setting, and `--assistant-mask-mode` as training. Its
primary metric is:

```text
sum(masked causal cross entropy) / number of positive-weight target tokens
```

The report also shows perplexity (`exp(CE/token)`) and the exact held-out target
token count. Every artifact must produce the same count as the base model.

Held-out CE in `model.eval()` has no Monte Carlo draw, so one evaluation pass
is sufficient for an artifact. Optimization still varies with LoRA
initialization, data order, and dropout, so the benchmark runs three training
seeds by default and reports mean and standard deviation.

## Workload Controls

Tokenization, masking, and optimization choices define the workload profile.
They must remain fixed across every item in one result set.

| control | accepted value | benchmark meaning |
|---|---|---|
| `--model` | model id or alias | Fixes the causal model, tokenizer, chat template, and supported LoRA layout. |
| `--data` | local path, Hugging Face id, or S3 prefix | Supplies `messages` rows and optional `tools`. |
| `--seq-len` | positive integer | Fixes packed sequence length and raw tokens per micro batch. |
| `--train-on` | `assistant` or `all` | Defines which causal targets receive positive loss weight. |
| `--assistant-mask-mode` | `native` or `legacy` | Fixes how assistant targets are derived. `native` requires the selected model chat template to expose exact `{% generation %}` spans; `legacy` is the explicit synthetic-format compatibility path. |
| `--token-budget` | positive integer | Requested global number of packed input tokens per item. |
| `--eval-rows` | positive integer | Reserves the final rows as the fixed held-out set. |
| `--seeds` | one or more training seeds | Repeat full training to estimate optimization variance. |
| batch recipe | micro batch and gradient accumulation | Fixes per-rank work and contributes to the optimizer-step count. |
| optimizer recipe | inner LR, weight decay, and warmup steps | Must be identical for an arm and its matching baseline. |
| LoRA recipe | rank, alpha, and target selection | Fixes the trainable layout and synchronized payload. |
| execution recipe | `--shard`, `--learner-gpus`, gradient checkpointing, and device choices | Fixes ranks per island, memory behavior, and hardware placement. |

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
the actual production sequence length, chat template, `--train-on` mode,
assistant-mask mode, LoRA targets, and batch recipe. Compare arms only within
a profile.

## Benchmark Items

Every reported row belongs to one of three item types. Only rows with the same
`M` and training seed form a quality pair.

| item | execution | artifact evaluated | purpose |
|---|---|---|---|
| `base` | no training | unmodified base model | Establish the pre-training loss and verify that fine-tuning helps. |
| `baseline-mM` | one synchronous process group with `M * G` ranks | rank-0 adapter after the final optimizer step | Supply the equal-hardware reference for every arm with that `M`. |
| DiLoCo arm | `M` independent process groups with `G` ranks each | adapter exported from the Rust syncer checkpoint | Isolate one synchronization choice while keeping the workload fixed. |

### Common DiLoCo Configuration

Unless an arm overrides a field, the harness uses the following synchronization
configuration. `--fragments` replaces `P=8` for every selected arm.

| field | default | meaning |
|---|---:|---|
| islands (`M`) | `2` | Independent learner process groups. |
| fragments (`P`) | `8` | Trainable adapter layout partitions synchronized independently. |
| fragment pattern | `binpack` | Size-balanced parameter grouping. |
| matrix merge | `rda` | Weighted radial-directional aggregation for non-embedding tensors. |
| broadcast blend (`alpha`) | `0.5` | Apply `0.5 * local + 0.5 * global` when a fragment returns. |
| learner push dtype | `bf16` | Learner deltas sent without q4 compression. |
| pipeline depth | `2` | Up to two distinct fragment rounds in flight. |
| delta correction | `heloco` | Correct stale directions before aggregation. |
| quorum | all `M` learners | Require every island for each completed fragment round. |
| outer learning rate | `0.7` | Scale applied by the syncer's outer optimizer. |
| outer momentum | `0.9` | Nesterov momentum held by the syncer. |
| target sync interval (`H`) | `24` inner steps | Throttle fast local rounds toward the production cadence. |

### Arm Matrix

The table lists every preset accepted by `--settings`. Unlisted fields retain
the common values above.

| arm | `M` | exact override | isolated question | required interpretation |
|---|---:|---|---|---|
| `m2` | 2 | none | Does the production-shaped two-island path match sync? | Compare only with `baseline-m2`. |
| `m4` | 4 | `learners=4` | Does quality remain stable with more independent islands? | Compare only with `baseline-m4`; total ranks remain `4 * G`. |
| `alpha0` | 2 | `merge_alpha=0.0` | Does overwriting with the returned global fragment outperform local/global blending? | A change isolates broadcast application, not aggregation. |
| `q4` | 2 | `wire_dtype=q4` | What quality cost accompanies lower learner egress? | Only learner pushes are q4; initialization and broadcasts remain bf16. |
| `serial` | 2 | `pipeline=1` | Does allowing two in-flight fragment rounds introduce harmful staleness? | Lower concurrency may change wall time even when quality is unchanged. |
| `noheloco` | 2 | `delta_correction=none` | Is HeLoCo correcting useful stale directions? | The merge rule and outer optimizer remain unchanged. |
| `strided` | 2 | `fragment_pattern=strided` | Does depth-interleaved fragment construction affect quality? | Fragment count stays fixed; only parameter-to-fragment assignment changes. |
| `iso` | 2 | `matrix_merge=iso` | Does Iso-C-style spectrum flattening improve matrix aggregation? | Matrix deltas use isotropic aggregation; unsupported tensor shapes retain direct averaging. |
| `direct-rda` | 2 | `outer_lr=1.0`, `outer_momentum=0.0`, `merge_alpha=0.0` | Is outer-optimizer gain or broadcast blending causing the quality gap? | This is direct application of the merged RDA delta, not plain parameter averaging. |
| `unthrottled` | 2 | `sync_interval_steps=0.0` | How sensitive is training to low-latency over-synchronization? | This is a localhost/LAN stress arm; use measured `H`, not the preset name, to confirm cadence. |

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

The generated report fields have the following meanings:

| field | definition | use |
|---|---|---|
| `raw tokens` | Actual packed tokens processed after step-count ceiling. | Confirm equal compute budget within a pair. |
| `target tokens` | Positive-weight causal targets that enter CE. | Reject assistant-mask accounting mismatches. |
| `target %` | `target tokens / raw tokens`. | Characterize how much packed context contributes optimization targets. |
| `CE/token` | Total masked held-out CE divided by held-out target tokens. | Primary quality metric; lower is better. |
| `perplexity` | `exp(CE/token)`. | Readable transform of the same quality metric, not an independent score. |
| `delta vs sync` | Per-seed percentage CE difference from `baseline-mM`, then mean and standard deviation. | Primary DiLoCo gap; lower is better and negative means improvement. |
| `train s` | Learner training wall time for the item. | Compare runtime only at matching `M` and hardware. |
| `raw tok/s` | Actual raw tokens divided by training wall time. | Workload throughput. |
| `target tok/s` | Actual target tokens divided by training wall time. | Effective supervised-target throughput. |
| `GPU-h` / `cost` | GPU count times wall time, optionally multiplied by `--gpu-hour-cost`. | Resource accounting, not a quality measure. |
| `mean H` | Mean learner inner-step count attached to accepted fragment responses. | Verify the realized synchronization cadence. |
| `tokens/response` | Mean raw-token progress represented by one accepted fragment response. | LM-specific cadence in workload units. |
| `participation` | Accepted responses divided by the maximum possible responses. | Detect missing or consistently late islands. |
| `stale` | Mean difference between current and response base fragment versions. | Detect stale updates hidden by aggregate loss. |
| `sync GB` | Estimated accepted tensor payload for initialization, pushes, and broadcasts. | Compare protocol payloads; it excludes framing, retries, and TCP overhead. |

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

| output | contents | role |
|---|---|---|
| `config.json` | Arguments, selected arm definitions, fairness contract, resume identity, and data manifest. | Audit the exact experiment plan. |
| `results.jsonl` | One durable record per completed `(kind, arm, seed)` run. | Resume unit and source record for aggregation. |
| `summary.json` | Cross-seed means, standard deviations, and paired deltas. | Machine-readable benchmark conclusion. |
| `report.md` | Human-readable quality and systems tables. | Review and publication surface. |
| `--work-dir` logs | Learner, baseline, syncer, export, and evaluation logs plus event tapes and checkpoints. | Diagnose failed or invalid items. |

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

## Current Scope

The harness covers LoRA causal-LM fine-tuning with masked cross entropy. Full
parameter synchronization, multi-node launch orchestration, WAN fault
injection, task-suite evaluation, and checkpoint-time learning curves are
separate experiments.

## Results

Completed Qwen3.6 benchmark results, including aggregate and per-seed
quality, execution, and synchronization tables, are archived in
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md#qwen36-27b-lm).
