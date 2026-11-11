# Kimi K3.1 full-parameter trainer: 128-expert layout on 6 x 8 B300

This document describes the recommended trainer layout for full-parameter RL
on Kimi-K3.1-Pro with 128 trainable experts per MoE layer. It is the single
reference for this layout. The end-to-end guide
([KIMI_K31_ASYNC_RL_GUIDE.md](KIMI_K31_ASYNC_RL_GUIDE.md), section 10) links
here.

## Validation status

| Configuration | Status |
|---|---|
| 64-expert full-parameter, 2 x 8 B300, end to end (NC samplers, Modal trainer) | Validated on hardware |
| 128-expert, 6 x 8 B300, **524k** total context, PP split **32/28/33** | Measured on hardware: one full training step with the stage-0 gradient fix active |
| 128-expert, 6 x 8 B300, **262k** total context, PP split **32/29/32** | Recommended production setting. Derived from a per-stage memory model fitted to the 524k measurement |
| 128-expert, 6 x 8 B300, 1M total context | Current scope: does not fit with PP3 and no activation offload (measured) |

Numbers in this document are labelled **[measured]** or **[estimate]**.

## 1. Which "262k"?

The layout uses context parallelism CP=2, so each CP rank holds half of the
sequence.

| Name | `--seq-len` (total) | `--rl-max-tokens-per-gpu` (per CP rank) | Status |
|---|---|---|---|
| 262k (recommended) | 262144 | 131072 | [estimate] |
| 524k (measured upper bound) | 524288 | 262144 | [measured] |
| 1M | 1048576 | 524288 | Does not fit [measured] |

## 2. Scope

- Model: `moonshotai/Kimi-K3.1-Pro`. It has 93 decoder layers, 92 of them
  MoE (HF layers 1..92), with 896 routed experts per layer and top-16 routing.
- Training form: 128 uniformly spaced trainable routed experts per MoE layer
  (about 389B trainable parameters). All other parameters are frozen in
  MXFP4. The run uses routed-expert MXFP4 QAT, full activation recompute, R3
  routing replay, and the score_centering loss over the rollout top-128
  log-probs.
- Hardware: Modal clustered, 6 nodes x 8 B300 = 48 GPUs, with about 267.7 GiB
  visible to torch per GPU. Inter-node NCCL runs over IB/RoCE. Each container
  has 64 vCPUs (the Modal per-function limit) and requests 2400 GiB of memory.

## 3. Expert selection file

The selection file is JSON, `schema: 1`, with 92 layer keys `"1"`..`"92"`:

```json
{
  "schema": 1, "model": "kimi_k31", "num_experts": 896, "num_trainable": 128,
  "ep_size": 16,
  "layers": { "1": { "trainable": [0, 7, 14, "... 128 ids"], "slot_order": ["... 896 ids"] }, "...": {} },
  "source": { "method": "uniform", "tokens": 0 },
  "sha256": "<canonical sha, covers slot_order>",
  "trainable_sha256": "<EP-independent sha of the trainable sets>"
}
```

`slot_order` depends on `ep_size`. This layout runs EP16, so the file must be
built with `--ep-size 16`. A file built for another EP size is refused at
load (`slot_order was built for ep_size 8, the run uses 16`).

```bash
# from scratch
python -m yeto.rl.adapters.miles.models.partial_experts.selection --model kimi_k31 \
  --num-experts 896 --num-trainable 128 --ep-size 16 --moe-layer-offset 1 \
  --mode uniform --num-moe-layers 92 --output <selection.ep16.json>
# from a file built for another EP size (same trainable set, same trainable_sha256)
python -m yeto.rl.adapters.miles.models.partial_experts.selection \
  --input <selection.ep8.json> --ep-size 16 --output <selection.ep16.json>
python -m yeto.rl.adapters.miles.models.partial_experts.selection --validate <selection.ep16.json>
```

## 4. Layout

| Item | Value |
|---|---|
| Parallelism | PP3 x TP8 x CP2 x EP16, DP1, sequence parallel on. TP groups span nodes: pass `--rl-allow-cross-node-tp`. |
| PP split, 262k | **32 / 29 / 32** [estimate]: `--decoder-first-pipeline-num-layers 32 --decoder-last-pipeline-num-layers 32` |
| PP split, 524k | **32 / 28 / 33** [measured]: `--decoder-first-pipeline-num-layers 32 --decoder-last-pipeline-num-layers 33` |
| Tokens per CP rank | `--rl-max-tokens-per-gpu = seq-len / CP` (131072 at 262k). A full-length sample then never splits across microbatches. A smaller value is refused. |
| Microbatches per step (m) | At least PP (3). Read the actual value from the `num_microbatches=[...]` log line. Packing short traces lowers m. |
| Distributed timeout | `--rl-distributed-timeout-minutes 60` at 262k, `120` at 524k. The first step's JIT plus long-sequence pipeline traffic took 662 s [measured], longer than the default. |

The split flags reach Miles through `YETO_MILES_TRAIN_EXTRA_ARGV`. The
launcher validates the split with `check_pipeline_split` before any GPU
starts (`yeto launch ... --dry-run` shows the result).

Why the split is uneven: under 1F1B, stage 0 holds the most in-flight
microbatches and is the tightest stage. A split such as 31/29/33 moves the
pressure to stage 1 and is not recommended.

## 5. Software

### 5.1 Trainer image

Use the full-parameter default image:

```
docker:ghcr.io/agentenv/yeto-miles-k31@sha256:2de5d796de19d38f4e3165efce2a9528a7f7d3e60e3cb32fe29fd00c6abf9b5b
```

It carries Miles `k31/integ-trainer` `1f29e43043`, which contains every Miles
change this layout depends on:

| Miles change | Purpose | Setting |
|---|---|---|
| DSA indexer scored in query blocks (`c5393bbeb4`) | Keeps the frozen indexer's score matrix bounded at long context | `MILES_K31_INDEXER_QUERY_CHUNK` (default 8192) |
| Attention-residual aggregation in token blocks (`a0cef1d401`) | Bounds the fp32 temporaries at 262k per rank | `MILES_K3_ATTN_RES_TOKEN_CHUNK` (default 4096) |
| Stage-0 expert gradients under full recompute (`2fc2b14adc`) | Ensures the trainable experts on PP stage 0 receive gradients | built in |
| Finite score-centering loss at head mass >= 1 (`0cb40f2fcf`) | Keeps loss and gradients finite | built in |

The 524k measurement ran with these same Miles changes applied on an earlier
base image. The full-parameter image above is the one this repository pins
and the one the validated Modal end-to-end run used.

### 5.2 yeto

This repository includes the stage-0 gradient hook
(`yeto.rl.adapters.miles.models.kimi_k31.first_stage_grad.before_train_step`)
and the uneven PP split support. Recommended practice: keep the hook enabled
as a second safeguard for stage-0 gradients, as in the measured run.

### 5.3 Environment

| Setting | Measured (524k) | Recommended (262k) |
|---|---|---|
| `MILES_K31_INDEXER_QUERY_CHUNK` | 2048 | 1024 [estimate] |
| `MILES_K3_ATTN_RES_TOKEN_CHUNK` | 4096 | 2048 [estimate] |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | on | off: a colocated Miles island launched through yeto does not accept it (torch_memory_saver) [estimate] |
| `NCCL_RAS_ENABLE` | 0 | 0 |
| `OMP_NUM_THREADS` | 8 | 8 |
| `MILES_MXFP4_QAT_TRITON` | 1 | 1 |
| `--log-probs-chunk-size` (Miles) | 4096 | 16384; lower to 4096 if the log-prob phase runs out of memory |
| Optimizer | CPU offload 1.0 | CPU offload 1.0 |

## 6. Launch command

The trainer-island part of `trainer.argv`. All site values are placeholders.

```bash
SEQ=262144; CP=2; MTPG=$((SEQ / CP))   # 524k: SEQ=524288, last stage 33, timeout 120

yeto launch --training-mode rl \
  --gpu modal:6x8xb300 --cluster-prefix <run-name> \
  --rl-image "docker:ghcr.io/agentenv/yeto-miles-k31@sha256:2de5d796de19d38f4e3165efce2a9528a7f7d3e60e3cb32fe29fd00c6abf9b5b" \
  --model moonshotai/Kimi-K3.1-Pro --model-revision <revision> \
  --modal-model-volume <weights-volume> \
  --data <prompt-dataset> --data-revision <revision> \
  --rl-prompt-column <prompt-col> --rl-label-column <label-col> \
  --reward-function <module:function> \
  --trust-remote-code --tuning full --rl-model-recipe kimi_k31 \
  --rl-expert-selection <selection.uniform128.ep16.json> --rl-routed-expert-qat mxfp4 \
  --expert-parallel 16 --tensor-parallel 8 --pipeline-parallel 3 --context-parallel 2 \
  --rl-allow-cross-node-tp \
  --rl-sample-role trainer --rl-single-island-no-sync --controller local \
  --rl-buffer-shards <shard-base>-0,<shard-base>-1,<shard-base>-2 --rl-stall-timeout 7200 \
  --rl-checkpoint-store s3://<bucket>/runs/<run-name> \
  --rl-object-store-endpoint <store-endpoint> \
  --modal-store-secret <modal-secret> \
  --inner-lr 1e-4 --rl-lr-schedule constant \
  --rl-max-policy-age 12 --rl-algorithm-spec <algorithm-spec.json> \
  --rl-release-every 2 --rl-release-skip-busy \
  --rl-loss score_centering --rl-score-centering-is tis --rl-score-centering-tis-clip 2.0 \
  --rl-rollout-top-logprobs-num 128 \
  --rl-allow-unverified-mechanism losses:score_centering \
  --rl-allow-unverified-mechanism features:rollout_logprobs_as_old \
  --rl-allow-unverified-mechanism features:no_grpo_std_normalization \
  --rl-allow-unverified-mechanism loss_aggregations:token \
  --rollout-batch-size 8 --n-samples-per-prompt 8 \
  --rollout-max-response-len $((SEQ - 16384)) --seq-len $SEQ \
  --rl-activation-recompute --rl-max-tokens-per-gpu $MTPG \
  --rl-distributed-timeout-minutes 60 \
  --rl-optimizer-offload-fraction 1.0 --modal-memory-gib 2400 --modal-cpu 64 \
  --modal-env OMP_NUM_THREADS=8 --modal-env MILES_MXFP4_QAT_TRITON=1 \
  --modal-env NCCL_RAS_ENABLE=0 \
  --modal-env MILES_K31_INDEXER_QUERY_CHUNK=1024 \
  --modal-env MILES_K3_ATTN_RES_TOKEN_CHUNK=2048 \
  --modal-env "YETO_MILES_TRAIN_EXTRA_ARGV=--log-probs-chunk-size 16384 --no-check-for-nan-in-loss-and-grad --custom-megatron-before-train-step-hook-path yeto.rl.adapters.miles.models.kimi_k31.first_stage_grad.before_train_step --decoder-first-pipeline-num-layers 32 --decoder-last-pipeline-num-layers 32" \
  --modal-env YETO_RL_NAN_SKIP=max_consecutive=2,max_fraction=0.05,min_rounds=20 \
  --total-steps <N> --rl-stop-after-rounds <N> --rl-cut-every <N>
```

Notes:

- Run it with `--dry-run` first (`yeto run up --dry-run` does this). The dry
  run checks the selection against `--expert-parallel` and the PP split, and
  prints the learner command.
- `--no-check-for-nan-in-loss-and-grad` with `YETO_RL_NAN_SKIP` turns a
  non-finite step into a counted skip. Remove both for strict behaviour.
- The algorithm values (learning rate, policy age, release cadence, TIS clip)
  are the formal-run defaults, not layout requirements.

## 7. Measured data (6 x 8 B300)

Synthetic long-context rollouts replayed through the Miles train loop. Length
is per CP rank. Memory is torch `max_memory_allocated / max_memory_reserved`
in GiB per stage.

| Round | Length / rank (total) | m | PP split | Stage-0 fix | step s | tok/s | s0 | s1 | s2 |
|---|---|---|---|---|---|---|---|---|---|
| **upper bound** | **262k (524k)** | **4** | **32/28/33** | **on** | **1019.8** | **2423** | **224.5 / 241.2** | **209.5 / 231.1** | **209.6 / 232.4** |
| 1M | 524k (1M) | 4 | 40/28/25 | on | out of memory on s0 | | | | |

- At 524k, nvidia-smi peaks were about 269.8k / 259.8k / 261.3k MiB per
  stage. Non-torch overhead was about 21-22 GiB per GPU, and stage 0 had about
  4 GiB of headroom.
- Numerics: all 92 MoE layers received non-zero gradients, and no step
  produced NaN. Chunked and unchunked runs gave a bit-identical loss at 131k
  per rank.
- Initialisation took about 19 minutes (1125.6 s) from launch to first
  generate. Budget it on every relaunch.
- Earlier rounds at m=1 (31/31/31, 38/29/26) understate memory and time and
  are not a basis for sizing.

## 8. Current scope and recommended practice

**Current scope**

- The 262k split 32/29/32 comes from a per-stage memory model:
  peak = 2.8 GiB x layers + k x 0.4375 GiB x sum(1 + rows) + about 30 GiB,
  with k = 3/2/1 in-flight microbatches on s0/s1/s2. The model matched the
  524k run within about 5 GiB. It predicts about 67 GiB of headroom at 262k
  [estimate].
- The recommended 262k settings differ from the measured run in three ways:
  expandable segments off, smaller chunk sizes, and log-prob chunk 16384.
- With real sampler buffers, the 128-expert path has completed
  initialisation and reached generation. A first optimizer step on that path
  is the next validation milestone.
- 1M total context needs about 456 GiB on stage 0 against about 268 GiB
  available. It requires a different layout (deeper PP or activation offload).
- Optimizer CPU offload takes a noticeable share of short test steps. At
  production batch sizes it is estimated at 2-5% [estimate].
- A tensor-parallel split of the DSA indexer exists in Miles
  (`MILES_K31_INDEXER_TP_SPLIT=1`, off by default). It is not yet validated
  for production.

**Recommended practice**

- Build the selection file with `--ep-size` equal to `--expert-parallel` (16).
- Keep `--rl-max-tokens-per-gpu` at `seq-len / CP`, and keep m >= PP.
- Set `--rl-distributed-timeout-minutes` (60 at 262k, 120 at 524k).
- Keep the stage-0 gradient hook in `YETO_MILES_TRAIN_EXTRA_ARGV`.
- Use learning rates around 1e-4 for the full-parameter form. The MXFP4
  release rounds to nearest, so very small updates (around 1e-6) can leave the
  released weights unchanged.
- Do not enable debug probes (per-layer NaN probes allocate extra memory).
- Request `--modal-cpu 64`; higher values are refused by Modal.

## 9. Validating on your cluster

Run these in order and stop at the first failure.

1. **Dry run.** It must exit 0, report 48 GPUs, and show the split flags and
   the stage-0 hook in `YETO_MILES_TRAIN_EXTRA_ARGV`.
2. **First step memory.** Record torch `max_memory_reserved` /
   `max_memory_allocated` for one rank per stage, and sample nvidia-smi
   `memory.used`. Treat less than 15 GiB of nvidia-smi headroom on any stage
   as a stop signal. Confirm `num_microbatches=[m]` with m >= 3.
3. **Stage-0 gradients.** After step 1, check that the trainable experts of
   stage-0 layers (for example HF layers 1, 10, 30) have non-zero gradient
   norms, like layers on later stages.
4. **Numerics.** Loss and grad_norm are finite, and
   `train_rollout_logprob_abs_diff` stays at your sampler's noise floor.
5. **Time.** Discard step 1 (JIT and autotune). Record `perf/step_time` and
   `perf/actor_train_time` from step 2.

**If a stage runs out of memory**, adjust in this order:

1. Identify the stage from the GPU in the error message.
2. Stage 0: lower `MILES_K31_INDEXER_QUERY_CHUNK` (2048, 1024, 512) and
   `MILES_K3_ATTN_RES_TOKEN_CHUNK` (4096, 2048, 1024), then move one layer off
   stage 0 (for example 31/30/32). Each layer is about 2.8 GiB.
3. Stage 1 or 2: move a layer toward stage 0 if it has headroom, or lower the
   chunk sizes.
4. Log-prob phase: lower `--log-probs-chunk-size` to 4096.
5. Large "reserved but unallocated" memory indicates fragmentation; reduce
   the chunk sizes.
6. If memory is still short, use a shorter context.
