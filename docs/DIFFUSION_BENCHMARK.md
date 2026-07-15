# Diffusion DiLoCo Benchmark

## Question

The benchmark tests one product claim:

> At the same model, data, LoRA recipe, device count, and training-sample
> budget, splitting the devices into independent DiLoCo islands should produce
> held-out diffusion loss close to one synchronous process group.

It is a quality and systems-ablation benchmark. It is not a model leaderboard,
an image/video aesthetic evaluation, or a cloud provisioning benchmark.

The executable harness is `scripts/benchmark_diffusion_diloco.py`.
The corresponding causal-LM design and a project-by-project comparison are
in `docs/LM_BENCHMARK.md`.

## Fairness Contract

For an arm with `M` islands and `G` ranks per island:

- the synchronous baseline uses one process group with `M * G` ranks;
- the DiLoCo arm uses `M` process groups with `G` ranks each;
- both use the same per-rank micro batch and gradient accumulation;
- both run the same number of optimizer steps;
- both therefore process the same total number of training examples;
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

## Arms

| arm | change from production M=2 | question |
|---|---|---|
| `m2` | none | Does the production-shaped two-island path match sync? |
| `m4` | four islands | How does quality scale with more independent islands? |
| `alpha0` | overwrite broadcasts | Is local/global blending helping? |
| `q4` | q4 learner pushes | What quality cost buys the WAN reduction? |
| `serial` | one fragment round in flight | Does pipelining add harmful staleness? |
| `noheloco` | no delta correction | Is HeLoCo correcting useful stale directions? |
| `strided` | depth-interleaved fragments | Does fragment construction affect quality? |
| `direct-rda` | no Nesterov, full outer step, overwrite | Is outer-optimizer gain causing a gap? |
| `unthrottled` | no H target | How sensitive is diffusion to low-latency over-syncing? |

`direct-rda` is deliberately not called `avg`. Non-embedding fragments still
use radial-directional averaging; the arm isolates Nesterov gain and broadcast
blending without claiming to change the merge operator.

The ordinary `m2` and `m4` arms retain the production `H=24` target.
`unthrottled` is the explicit localhost/LAN stress arm.

## Systems Measurements

Held-out loss answers the quality question, but a run is only interpretable if
the intended synchronization actually happened. The report also records:

- measured mean inner steps per fragment response (`H`);
- responder participation rate;
- mean fragment-version staleness;
- merge and sync duration counts;
- estimated accepted raw tensor bytes for initialization, pushes, and broadcasts;
- wall time, samples/second, GPU-hours, and optional estimated cost.

These are diagnostics, not substitutes for held-out quality. In particular,
localhost wall time does not measure cross-region TCP throughput.

## Reproducibility

The benchmark uses explicit micro batch sizes rather than autobatching and
defaults DataLoader workers to zero. Each training seed controls LoRA
initialization, data order, timestep sampling, and noise through the diffusion
learner's seeded RNG streams. Evaluation uses a separate fixed seed so every
artifact sees identical Monte Carlo samples.

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

- `config.json`: full arguments, arms, and fairness contract;
- `results.jsonl`: one record per seed and arm;
- `summary.json`: cross-seed aggregates;
- `report.md`: human-readable comparison table;
- per-run learner, syncer, export, and evaluation logs under `--work-dir`.

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
