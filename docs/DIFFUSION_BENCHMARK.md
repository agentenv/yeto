# Diffusion DiLoCo Benchmark

## Question

The benchmark tests one product claim:

> At the same model, data, LoRA recipe, device count, and training-sample
> budget, splitting the devices into independent DiLoCo islands should produce
> held-out diffusion loss close to one synchronous process group.

It is a quality and systems-ablation benchmark. It is not a model leaderboard,
an image/video aesthetic evaluation, or a cloud provisioning benchmark.

The executable harness is `scripts/benchmark_diffusion_diloco.py`.

## Fairness Contract

For an arm with `M` islands and `G` ranks per island:

- the synchronous baseline uses one process group with `M * G` ranks;
- the DiLoCo arm uses `M` process groups with `G` ranks each;
- both use the same per-rank micro batch and gradient accumulation;
- both run the same number of optimizer steps;
- both therefore process the same total number of training examples;
- corresponding logical ranks consume the same row streams and use the same
  training RNG streams for timestep, noise, and other stochastic operations;
- model id, dataset, LoRA rank/alpha/targets, optimizer, loss weighting,
  spatial/video shape, and root training seed are identical;
- all runs use the same held-out rows and paired evaluation RNG draws.

The diffusion learner rebalances explicit accumulation as
`ceil(requested_grad_accum / micro_batch)`. The benchmark uses that effective
value in all budget accounting. The step count is:

```text
ceil(sample_budget / (micro_batch * effective_grad_accum * M * G))
```

The ceiling can overshoot the requested sample budget by less than one global
parallel step. The report records the actual processed count. Comparisons are
always arm-to-its-matching-baseline at the same `M`; different `M` values may
have slightly different ceiling overshoot.

Unlike a one-learner baseline, this topology keeps total hardware and global
batch size equal. Wall time, samples/second, and GPU-hours are therefore
comparable within a given `M`.

## Authoritative Artifact

A DiLoCo learner applies delayed broadcasts with local blending, so its local
adapter is not the global result. The benchmark never scores a learner output.
It checkpoints the real Rust syncer, rebuilds the deterministic trainable
layout with `yeto.diffusion.export`, and evaluates the exported merged adapter.

This matches the artifact Yeto can recover after every learner has disappeared.

## Data And Evaluation

The final `--eval-rows` rows are held out before training. Every arm trains on
the same preceding rows. Local relative media and cache paths are rewritten to
absolute paths before subsets are saved.

Diffusion loss is stochastic, so one evaluation pass is not enough. For every
held-out row and repetition, the harness derives the same isolated RNG seed for
every artifact. This pairs timestep and noise draws across the base model,
synchronous baseline, and all DiLoCo arms.

The primary metric is:

```text
sum(flow_matching_loss) / sum(predicted_elements)
```

It is only comparable within the same model, scheduler, loss weighting, and
shape. It is not comparable across unrelated diffusion families.

At least three training seeds are used by default. The report shows mean and
standard deviation of final held-out loss and percentage delta against the
matching synchronous baseline.

## Workload Controls

Shape and preprocessing choices define the workload profile. They must remain
fixed across every item in one result set.

| control | accepted value | benchmark meaning |
|---|---|---|
| `--model` | model id or alias | Fixes the denoiser, scheduler, latent geometry, and supported adapter path. |
| `--data` | local path, Hugging Face id, or S3 prefix | Supplies prompt plus image, video, or cached latent rows. |
| `--height`, `--width` | required positive integers | Fix the spatial training and evaluation shape. |
| `--resize-mode` | `stretch` or `center-crop` | Fixes how raw media is transformed to the requested shape. |
| `--num-frames`, `--fps` | optional video controls | Fix temporal length and sampling rate for video workloads. |
| cache flags | `--cache-latents`, `--cache-text-embeds` | Select raw-media/raw-prompt processing or precomputed tensors; the choice must not vary by item. |
| column flags | image, video, prompt, latent, text embedding, mask, and pooled embedding columns | Bind dataset schema explicitly. |
| `--diffusion-loss-weighting` | `none`, `linear`, `sigma`, `snr`, or `min-snr` | Fixes weighting over sampled timesteps; `--diffusion-min-snr-gamma` applies to `min-snr`. |
| `--sample-budget` | positive integer | Requested global number of training examples per item. |
| `--eval-rows`, `--eval-repeats`, `--eval-seed` | positive rows/repeats and one fixed seed | Define the held-out Monte Carlo evaluation set. |
| `--seeds` | one or more training seeds | Repeat full training to estimate optimization variance. |
| batch and LoRA recipe | micro batch, grad accumulation, LR, weight decay, warmup, LoRA rank/alpha/targets | Must be identical for an arm and its matching baseline. |

## Benchmark Items

Every reported row belongs to one of three item types. Only rows with the same
`M` and training seed form a quality pair.

| item | execution | artifact evaluated | purpose |
|---|---|---|---|
| `base` | no training | unmodified base pipeline | Establish the pre-training held-out loss and verify that fine-tuning helps. |
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
| merge rule | `rda` | Weighted radial-directional aggregation for non-embedding tensors. |
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
| `direct-rda` | 2 | `outer_lr=1.0`, `outer_momentum=0.0`, `merge_alpha=0.0` | Is outer-optimizer gain or broadcast blending causing the quality gap? | This is direct application of the merged RDA delta, not plain parameter averaging. |
| `unthrottled` | 2 | `sync_interval_steps=0.0` | How sensitive is training to low-latency over-synchronization? | This is a localhost/LAN stress arm; use measured `H`, not the preset name, to confirm cadence. |

## Systems Measurements

Held-out loss answers the quality question, but a run is only interpretable if
the intended synchronization actually happened. The report also records:

- measured mean inner steps per fragment response (`H`);
- responder participation rate;
- mean fragment-version staleness;
- merge and sync duration counts;
- estimated accepted raw tensor bytes for initialization, pushes, and broadcasts;
- wall time, samples/second, GPU-hours, and optional estimated cost.

The generated report fields have the following meanings:

| field | definition | use |
|---|---|---|
| `samples` | Actual examples processed after step-count ceiling. | Confirm equal training budget within a pair. |
| `loss/elem` | Total held-out flow-matching loss divided by predicted elements over all rows and repeats. | Primary quality metric; lower is better. |
| `delta vs sync` | Per-seed percentage loss difference from `baseline-mM`, then mean and standard deviation. | Primary DiLoCo gap; lower is better and negative means improvement. |
| `train s` | Learner training wall time for the item. | Compare runtime only at matching `M` and hardware. |
| `samples/s` | Actual processed samples divided by training wall time. | Workload throughput. |
| `GPU-h` / `cost` | GPU count times wall time, optionally multiplied by `--gpu-hour-cost`. | Resource accounting, not a quality measure. |
| `mean H` | Mean learner inner-step count attached to accepted fragment responses. | Verify the realized synchronization cadence. |
| `participation` | Accepted responses divided by the maximum possible responses. | Detect missing or consistently late islands. |
| `stale` | Mean difference between current and response base fragment versions. | Detect stale updates hidden by aggregate loss. |
| `sync GB` | Estimated accepted tensor payload for initialization, pushes, and broadcasts. | Compare protocol payloads; it excludes framing, retries, and TCP overhead. |

These are diagnostics, not substitutes for held-out quality. In particular,
localhost wall time does not measure cross-region TCP throughput.

## Reproducibility

The benchmark uses explicit micro batch sizes rather than autobatching and
requires DataLoader workers to remain zero. A DiLoCo rank `(learner_id, rank)`
is mapped to logical rank `learner_id + M * rank`, which is the matching rank in
`baseline-mM`. Each training seed therefore controls identical LoRA
initialization, data order, timestep sampling, noise, and other stochastic
operations on corresponding ranks. Evaluation uses a separate fixed seed so
every artifact sees identical Monte Carlo samples.

CUDA kernels are not forced into deterministic implementations. Repeated seeds
measure optimization variability, not bitwise reproducibility.

## Running

`--data` accepts the same local paths and Hugging Face dataset ids as the
learner, plus an S3 dataset prefix. S3 prefixes are downloaded read-only with
`aws s3 sync` into `<work-dir>/source-data` before the train/eval split is
materialized. The AWS CLI and ambient read credentials must be available on
the benchmark host. `--dry-run` validates the plan without downloading data.

Dry-run the topology and budgets without loading models or data:

```bash
python scripts/benchmark_diffusion_diloco.py \
  --model Lightricks/LTX-Video \
  --data /mnt/ltx-benchmark \
  --height 512 --width 512 --resize-mode center-crop \
  --num-frames 49 --fps 9.3 \
  --settings m2,m4,q4,unthrottled \
  --learner-gpus 2 \
  --dry-run
```

Run the benchmark:

```bash
python scripts/benchmark_diffusion_diloco.py \
  --model Lightricks/LTX-Video \
  --data s3://yeto-navalora/datasets/ltx_yeto_4352_compressed \
  --height 512 --width 512 --resize-mode center-crop \
  --num-frames 49 --fps 9.3 \
  --settings m2,q4,serial,noheloco,unthrottled \
  --learner-gpus 2 \
  --sample-budget 512 \
  --seeds 17,29,43 \
  --overwrite
```

Outputs:

| output | contents | role |
|---|---|---|
| `config.json` | Arguments, selected arm definitions, fairness contract, resume identity, and data manifest. | Audit the exact experiment plan. |
| `results.jsonl` | One durable record per completed `(kind, arm, seed)` run. | Resume unit and source record for aggregation. |
| `summary.json` | Cross-seed means, standard deviations, and paired deltas. | Machine-readable benchmark conclusion. |
| `report.md` | Human-readable quality and systems table. | Review and publication surface. |
| `--work-dir` logs | Learner, baseline, syncer, export, and evaluation logs plus event tapes and checkpoints. | Diagnose failed or invalid items. |

Use `--resume` after interruption. Resume verifies the original arguments, arm
definitions, implementation fingerprint, serialized split hashes, and local
source-data hash when applicable before reusing completed records and the
original train/eval splits. Legacy runs without an immutable resume manifest
must restart with `--overwrite`.

## Interpretation Rules

- Compare each DiLoCo arm only with `baseline-mM` for the same `M` and seed.
- Do not draw a conclusion from one training seed.
- Confirm that baseline training improves over the untrained base before
  interpreting DiLoCo deltas.
- Treat low participation or an unexpected measured `H` as an invalid run,
  not as an algorithm result.
- Treat q4 byte estimates as a lower-bound payload accounting, not a WAN
  throughput test; framing, retries, reconnects, and dropped pushes are omitted.
- Use downstream generation metrics only as a separate evaluator over the
  saved artifacts. They must not change the training or held-out-loss path.

## Current Scope

The harness intentionally covers LoRA and flow-matching loss, which are the
supported asynchronous diffusion path. Full-parameter FSDP synchronization,
cloud provisioning, FID/FVD/aesthetic metrics, checkpoint-time learning
curves, and forced learner failures are separate experiments.

## Results

Completed LTX-Video and Wan2.2 benchmark results, including aggregate and
per-seed quality, execution, and synchronization tables, are archived in
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md#ltx-video-diffusion) and
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md#wan22-t2v-a14b-diffusion).
