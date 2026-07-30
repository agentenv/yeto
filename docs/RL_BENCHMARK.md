# Miles RL LM Benchmark

`scripts/benchmark_rl.py` compares native Miles with the strict Yeto RL path
on one GPU host. It trains real causal language models with real Miles
rollouts and the selected reward callable; it does not inject trajectories,
use synthetic optimizer steps, or provision cloud resources.

## Comparison

For a benchmark size `M` and `G` GPUs per federated island, the harness runs
three arms:

| arm | topology | purpose |
| --- | --- | --- |
| `native-miles-mM` | one native Miles island on `M*G` GPUs | reference with optimizer state preserved across rounds and no Yeto policy hook |
| `yeto-single-mM` | one Yeto+Miles island on `M*G` GPUs | isolates the Yeto hook, strict checkpoint/apply contract, and LoRA optimizer reset |
| `yeto-federated-mM` | `M` Yeto+Miles islands on `G` GPUs each | measures fixed-roster exact-base averaging and the multi-island split |

`native -> yeto-single` measures the synchronization contract itself.
`yeto-single -> yeto-federated` measures island partitioning and averaging.
`native -> yeto-federated` is the end-to-end product comparison.

The single-island Yeto average is an identity operation on LoRA weights, but
it is not equivalent to native Miles: every committed global apply clears the
LoRA optimizer state while preserving LR scheduler progress.

## Work Accounting

Let `K` be `--groups-per-island`, `N` be `--samples-per-group`, and `R` be
`--global-rounds`.

- Each federated island processes `K` prompt groups per round; the native and
  Yeto-single arms process `M*K` groups per round.
- Every arm owns `M*G` GPUs and processes `R*M*K*N` trajectories.
- `K*N` must divide `optimizer_steps*G`, which gives every rank the same Miles
  batch for all three arms.
- Expert parallelism is one by default. An explicit `--expert-parallel` must
  divide `G` and is held fixed across all three arms, including MoE models.
- The maximum action-token budget is identical. Actual response tokens are
  recorded because learned policies can terminate at different lengths.

Training prompts are held in a round-major stream. The single-island arms see
the combined stream; federated island `i` sees its fixed slice from every
round. Rollout capture files are checked against those prompt identities
before a result is accepted. Every island also receives the same deterministic
sampling seed. Independent Miles/SGLang arms may still produce non-bitwise-
identical trajectories; the retained captures record the actual outputs.

The final rows of the source dataset are held out before any training stream
is built. Every artifact is evaluated on those rows with the same per-sample
generation seeds and the same Miles reward callable. The report includes mean
reward and standard pass@k estimates. A sample passes when its reward is
strictly greater than `--pass-threshold`.

## Running

Run inside the pinned Miles RL environment with a clean detached checkout at
the commit recorded by `yeto.rl`. The largest requested `M` needs `M*G` visible
GPUs. The harness starts one local Ray cluster per arm; concurrent federated
islands use independent placement groups and disjoint host-port ranges within
that cluster. `--miles-port-base` moves those ranges when the defaults conflict
with another local service.

```bash
python scripts/benchmark_rl.py \
  --model Qwen/Qwen3-4B \
  --model-revision <immutable-hf-commit> \
  --data <org/prompt-dataset> \
  --data-revision <immutable-hf-commit> \
  --reward-function project.rewards:score \
  --islands 2,4 \
  --gpus-per-island 2 \
  --global-rounds 8 \
  --groups-per-island 4 \
  --samples-per-group 4 \
  --eval-prompts 64 \
  --eval-samples-per-prompt 4 \
  --pass-k 1,4 \
  --trust-remote-code
```

With `G=2`, this example uses `4` total GPUs for every `M=2` arm and `8`
total GPUs for every `M=4` arm. Use `--dry-run` to inspect every topology and
work budget without importing Ray, loading data, or touching a model:

```bash
python scripts/benchmark_rl.py <same identity and workload arguments> --dry-run
```

Models must be Hugging Face repositories selected by an immutable commit;
mutable local model directories are rejected. Remote datasets likewise require
an immutable Hub revision. A local JSON/JSONL, Parquet, or
`datasets.save_to_disk` input is accepted without `--data-revision`; its
materialized files are content-hashed in the run manifest.

## Evaluation Scope

Held-out evaluation uses the pinned base model plus the standard PEFT adapter
in an independent process and, on CUDA, one visible GPU. The base model must
fit that evaluation device. It implements ordinary single-turn LM generation
with the model chat template. This is appropriate for the v0 LM benchmark,
but it does not claim to reproduce a custom multi-turn generate function or a
session-server environment. Such workloads need a benchmark built on the
corresponding Miles environment evaluator rather than this Transformers
generation path.

Yeto artifacts are always exported from the authoritative syncer checkpoint.
They are never taken from one island's local adapter. Native Miles uses its
own final LoRA save. The harness validates its complete tensor layout against
the model's PEFT contract and adds the PEFT wrapper prefix omitted by Miles;
tensor values are not changed. Miles performs the native save inside the
measured job, while Yeto checkpoint export follows the measured training job.
Results report both `train_wall_s` and the comparable `artifact_ready_s`; each
arm's adapter preparation duration is also retained as `artifact_s`.

## Outputs And Resume

The default output directories are `rl-benchmark-work/` and
`rl-benchmark-report/`.

- `config.json` contains the immutable workload, complete Yeto-source and
  built-syncer implementation fingerprint, Miles commit, reward and data
  hashes, and fairness contract.
- `results.jsonl` is updated atomically after each completed arm and seed.
- `summary.json` contains aggregates across training seeds.
- `report.md` contains quality deltas and systems measurements.
- Each arm directory retains Miles logs, real rollout captures, evaluation
  samples, and either the native adapter or authoritative Yeto export.

Use `--resume` with unchanged inputs and arguments to skip completed records.
The harness refuses changed data, implementation, arms, or workload settings.
Use `--overwrite` only when intentionally starting a new result set.

Systems fields include training and artifact-ready wall time, trajectories/s,
action tokens/s, GPU-hours, estimated cost, Yeto synchronization time and
bytes, and KL when emitted by the Yeto hook. Native Miles does not pass through
that hook, so hook-only diagnostics are intentionally absent for the native
arm.
