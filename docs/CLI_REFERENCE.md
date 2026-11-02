# CLI reference

This page documents the `yeto` command line in detail: launch concepts, every
subcommand with its common flags, the `yeto launch` flag groups, the other
console scripts, and running without a cloud. For an overview of the project,
see the [README](../README.md). Run any command with `--help` for the complete
option list.

## Contents

- [Key concepts](#key-concepts)
- [Commands](#commands)
- [Launch flags reference](#launch-flags-reference)
- [Running without a cloud](#running-without-a-cloud)

## Key concepts

### `--gpu` grammar

Comma-separated, one entry per learner island:

```
cloud:[NODESx]COUNTxGPU[@region]
```

| Entry | Meaning |
|---|---|
| `aws:8xa100@us-east-2` | one node with 8 A100s in `us-east-2` |
| `aws:4x8xa100@us-east-2` | one island of 4 nodes, 8 A100s each |
| `nebius:8xh100@eu-north1` | one Nebius node with 8 H100s |
| `aws:1xl4@us-west-2` | a single L4 |
| `modal:8xh100` | one Modal container with 8 H100s (no region pinning) |
| `modal:2x8xh100` | an island of two Modal containers |
| `ssh:1x8x910b4` | an existing Ascend node reached over SSH (RL, with `--rl-node-inventory`) |

The cloud prefix is any lowercase name. `modal` starts Modal containers,
`ssh` uses existing nodes from `--rl-node-inventory`, and every other prefix
(`aws`, `nebius`, `verda`, `runpod`, ...) is handed to SkyPilot, so it must be
a cloud that `sky check` reports as enabled. GPU names are case-insensitive (`a100`, `a100-80gb`, `a10g`, `l4`, `l40s`,
`h100`, `h200`, `b200`, `t4`, `v100`, `rtx-6000-ada`, `rtx-pro-6000`, Ascend
`910b`/`910b2`/`910b3`/`910b4`/`910c`, PPU `810e`/`m890`). Any other card
name is an error.

### `--data` formats

A Hugging Face dataset id, a local path (a jsonl, json
or parquet file, a directory of them, or a `save_to_disk` directory), or an
object-store URI (`s3://`, `gs://`, `r2://`, ...). Non-HF sources are shipped
to the learners through SkyPilot file mounts. Pin HF datasets and models with
`--data-revision` / `--model-revision`; both are resolved to an immutable
commit before launch. `--data-format auto` (default) recognizes:

```jsonl
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}
{"instruction": "Translate to French", "input": "cat", "output": "chat"}
```

(OpenAI `messages`, ShareGPT and Alpaca, respectively.) By default only
assistant tokens carry loss (`--train-on assistant`).

### `--output`

Any object-store URI SkyPilot supports (`s3://`, `gs://`,
`r2://`, `oci://`, ...) or `hf://org/repo`: the head uploads the final model
and then terminates itself. A local path, or no `--output`, keeps the artifact
on the head, which stays up until `yeto down`.

### QLoRA

NF4 frozen base with bitsandbytes on CUDA; requires DDP:

```bash
pip install -e ".[launcher,qlora]"
yeto launch --gpu aws:1xa100@us-east-2 --model llama31-8b --data org/chat-traces \
  --tuning lora --base-quantization nf4 --shard ddp
```

For models that do not fit on one GPU in bf16, use `--shard fsdp` (LoRA, shards
the frozen base) or `--island-backend megatron` with `--expert-parallel`,
`--tensor-parallel` and `--pipeline-parallel` (see
[docs/MEGATRON.md](MEGATRON.md)).

### Resume and branch

```bash
# continue training an earlier Yeto LoRA artifact with the same recipe
yeto launch --gpu aws:8xa100@us-east-2 --model qwen35-9b --data org/chat-traces \
  --tuning lora --lora-r 16 --lora-alpha 32 --seq-len 4096 \
  --resume-from s3://my-bucket/qwen35-9b-sft --adapter-sha256 <sha256>

# start a new run from any PEFT adapter, recording lineage
yeto launch --gpu aws:8xa100@us-east-2 --model qwen35-9b --data org/more-traces \
  --branch-from ./adapter
```

`--model` and `--data` are always required. `--resume-from` does not fill in
the recipe from the artifact; it checks it. The launch is refused unless the
dataset (source and resolved revision), `--trust-remote-code` and the training
recipe (`--tuning`, `--base-quantization`, `--shard`, `--lora-r`,
`--lora-alpha`, `--lora-targets`, `--loss-function`, `--train-on`,
`--assistant-mask-mode`, `--data-format`, `--seq-len`, `--micro-batch-size`,
`--grad-accum`, `--inner-lr`, ...) match the parent; duration, output and
infrastructure flags may change. Use `--branch-from` for any intentional
recipe change. The parent must be a local directory or an object-store URI
(not a Hub id); a cloud parent needs `--adapter-sha256`, and a local one is
hashed automatically. See
[docs/ADAPTER_LIFECYCLE.md](ADAPTER_LIFECYCLE.md).

### Merge for deployment

```bash
yeto merge --adapter-dir ./adapter --output-dir ./merged --max-shard-size 5GB
```

## Commands

Run any command with `--help` for its full option list.

### `yeto launch`

Submit a detached SFT or RL run. All flags are listed in
[Launch flags reference](#launch-flags-reference).

### `yeto shape`

Compute a fleet plan for a model under a budget and/or a TFLOPs target from
spot prices, quotas, capacity signals and a per-model memory estimate. Spot
quotas and spot placement scores are AWS-only signals; other clouds are
planned from catalog prices and their own capacity signals.

| Flag | Default | Description |
|---|---|---|
| `--model` | — | model alias or HF id |
| `--budget` | — | fleet budget in $/hr (includes the head VM) |
| `--flops` | — | target effective TFLOPs; minimize cost to reach it (combine with `--budget` for the cheapest plan under a cap) |
| `--tuning` | `lora` | `lora` or `full` |
| `--seq-len` | `2048` | sequence length used in the memory estimate |
| `--training-mode` | `sft` | `sft` or `rl`; `rl` prices fixed RL islands (actor + rollout GPUs) |
| `--parameter-mode` | `lora` | RL only: `lora` colocates rollout with the actor; `full` gives rollout its own GPUs |
| `--actor-gpus` | `8` | RL only: actor GPUs per node |
| `--actor-nodes` | `1` | RL only: nodes per island (lora mode) |
| `--rollout-num-gpus` | `0` | RL only: dedicated rollout GPUs (full mode) |
| `--data` | — | HF dataset id; fills the launch line, required with `--apply` |
| `--apply` | off | launch the computed plan immediately (hands off to `yeto launch`) |
| `--json` | off | print the plan as JSON |
| `--clouds` | — | clouds to plan across (default: aws plus every registered cloud with credentials) |
| `--regions` | — | `cloud:region` entries, e.g. `aws:us-east-1,nebius:eu-north1`; a bare region means aws; `cloud:all` lifts the limit for one cloud |
| `--gpus` | — | GPU allowlist in SkyPilot names, e.g. `A100,A100-80GB,H100` |
| `--max-islands` | `16` | cap on learner islands (syncer fan-out) |
| `--price-margin` | `0.15` | headroom on catalog spot prices when enforcing the budget |
| `--head-cost` | `0.4` | assumed $/hr of the head VM in budget math |
| `--min-score` | `7` | require a spot placement score strictly above this (0 keeps fetching scores but stops gating) |
| `--skip-capacity-check` | off | plan on quota and price alone, with no placement-score API calls |
| `--strict-capacity-check` | off | reject shapes whose placement score cannot be fetched |
| `--weights-gb` | — | override the model weight size estimate (bf16 GB) |
| `--no-cache` | off | bypass the 1-hour signal cache |

### `yeto status`, `logs`, `down`, `stop-run`

Inspect and end runs. Each takes the run name (its `--cluster-prefix`).

| Command | Purpose | Flags |
|---|---|---|
| `yeto status` | list runs | `--tape FILE`: also summarize a syncer event tape (JSONL) |
| `yeto logs RUN` | follow a run log | `--no-follow`: print what is captured so far and exit |
| `yeto down RUN` | tear down every cluster of a run | — |
| `yeto stop-run RUN` | stop a run cleanly | — |

### `yeto merge`

Merge a PEFT adapter into its base model and write SafeTensors shards.

| Flag | Default | Description |
|---|---|---|
| `--adapter-dir` | — | adapter directory |
| `--output-dir` | — | where the merged model is written |
| `--model` | — | base override; defaults to the base recorded in the artifact |
| `--model-revision` | — | base branch/tag/commit |
| `--trust-remote-code` | off | allow code from the model repository |
| `--device` | `cpu` | device used for the merge |
| `--dtype` | `auto` | `auto`, `bf16`, `fp16` or `f32` |
| `--max-shard-size` | `5GB` | maximum SafeTensors shard size, e.g. `2GB` or `500MB` |

### `yeto sample-diffusion`

Sample from a trained diffusion adapter on one cloud GPU.

```bash
yeto sample-diffusion --gpu aws:1xl4@us-west-2 --adapter-dir s3://my-bucket/flux-lora \
  --prompt "a watercolor fox" --num-inference-steps 30 --output ./samples
```

| Flag | Default | Description |
|---|---|---|
| `--gpu` | — | a single GPU cluster, e.g. `aws:1xt4@us-west-2` |
| `--adapter-dir` | — | local directory or cloud URI of a Yeto diffusion adapter |
| `--output` | — | local directory or URI for samples (default: under `~/yeto-output`) |
| `--prompt` | — | one prompt |
| `--data` | — | HF/local/cloud prompt dataset for batch sampling |
| `--prompt-column` / `--seed-column` | `prompt` / — | dataset columns |
| `--max-rows` | — | cap prompt rows |
| `--num-inference-steps` | `30` | denoising steps |
| `--guidance-scale` | — | classifier-free guidance scale |
| `--height`, `--width`, `--num-frames` | — | output size overrides |
| `--fps` | `8` | video frame rate |
| `--seed` | — | sampling seed |
| `--dtype` | `auto` | `auto`, `bf16`, `fp16`, `f32` |
| `--on-demand` | off | use on-demand instead of spot |
| `--disk-size` | `256` | sampler disk (GB) |
| `--cluster-prefix` | `yeto-sample` | cluster name prefix |
| `--keep` | off | do not tear down the sampler cluster |

It also accepts `--model`, `--model-revision`, `--data-revision`,
`--trust-remote-code`, `--diffusion-adapter`,
`--allow-unattested-legacy-adapter`, `--spot`, `--learner-cpus`,
`--learner-instance-type`, `--learner-image`, `--retry-until-up` and
`--controller-poll` with the same meaning as on `yeto launch`.

### `yeto dashboard`

A read-only RL fleet dashboard built from event tapes. See
[docs/DASHBOARD.md](DASHBOARD.md).

```bash
yeto dashboard serve --run rl-demo --port 8787       # local web UI (loopback only)
yeto dashboard export --run rl-demo -o report.html   # static HTML
yeto dashboard mirror --target user@host --remote /path/tape.jsonl --local ./tapes/
```

| Flag | Default | Applies to | Description |
|---|---|---|---|
| `--run` | — | serve, export | follow `~/.yeto/runs/<run>/` and the head's syncer tape |
| `--tapes` | — | serve, export | extra tape files or directories (`*.jsonl`, up to 2 levels deep) |
| `--prices` | — | serve, export | price table JSON (the built-in table is an example, not a bill) |
| `--budget` | — | serve, export | budget cap in $ for the cost bar and alerts |
| `--thresholds` | — | serve, export | alert threshold overrides JSON |
| `--compare` | — | serve, export | tapes of a second run to compare with |
| `--label`, `--compare-label` | — (shown as `A`, `B`) | serve, export | display labels for the comparison |
| `--host` | `127.0.0.1` | serve | loopback address only (`0.0.0.0` is refused) |
| `--port` | `8787` | serve | HTTP port |
| `--poll` | `2.0` | serve | tape follow interval (s) |
| `--head` | — | serve | head host name for the printed `ssh -L` hint |
| `-o`, `--out` | — | export | output `.html` |
| `--target`, `--remote`, `--local` | — | mirror | SSH target, remote tape path, local mirror path |
| `--island` | — | mirror | island id recorded on source-lost events |

### `yeto rl`

An experimental single-process PPO loop against a CyberGym server (or a mock
environment). See [docs/CYBERGYM_RL.md](CYBERGYM_RL.md).

| Flag | Default | Description |
|---|---|---|
| `--env` | `cybergym` | `cybergym` or `mock` |
| `--task` | `vulnerability_analysis` | task name |
| `--model` | `Qwen/Qwen2.5-0.5B` | policy model |
| `--iterations`, `--steps`, `--epochs`, `--batch-size` | `1`, `64`, `2`, `32` | loop sizes |
| `--lr`, `--gamma` | `1e-05`, `0.99` | optimizer LR and discount |
| `--budget` | `10.0` | printed at start for monitoring only; it does not limit the run |
| `--output` | — | output directory (default `./rl_output_<env>_<task>`) |
| `--server-host`, `--server-port` | `127.0.0.1`, `8666` | CyberGym server |

### Other console scripts

| Script | Purpose |
|---|---|
| `yeto-export` | export a syncer checkpoint as a Hugging Face model or adapter |
| `yeto-rl-export` | export a committed RL checkpoint as a standard PEFT adapter |
| `yeto-diffusion-export` | export a syncer checkpoint as a Diffusers/PEFT adapter directory |
| `yeto-diffusion-sample` | sample from a Yeto diffusion adapter artifact on the local machine |
| `yeto-protenix-export-batch`, `yeto-hunyuan3d-export-batch` | run inside a Protenix / Hunyuan3D checkout: export pre-collated training batches plus a JSONL manifest for Yeto diffusion training |
| `yeto-rl-ssh` | Miles RL on existing SSH hosts, with failure injection ([docs/RL_SSH_ACCEPTANCE.md](RL_SSH_ACCEPTANCE.md)) |
| `yeto-stage1` | expand a YAML run plan over an SSH node inventory (`plan`) and run it (`run`) |
| `yeto-codex-traces` | convert Codex ATIF trajectories or session logs to Yeto chat JSONL |
| `yeto-wandb-tape` | upload a syncer event tape (one JSONL record per merge) to W&B |

Module entry points: `python -m yeto.learner` and `python -m
yeto.mlx.learner` (manual learners, see below),
`python -m yeto.rl.engine.engine_sampler --spec SPEC.json` (one rollout island
from a sampler spec).

## Launch flags reference

`yeto launch --help` groups its flags as below. Defaults are the argparse
defaults; "—" means unset (the launcher or recipe picks a value, or the
feature is off). "off" is a boolean switch that is off by default.

### Options

The core of every launch: where to run, what to train and on which data.

| Flag | Default | Description |
|---|---|---|
| `--gpu` | — | learner clusters, `cloud:[NODESx]COUNTxGPU[@region]`, comma-separated; omit to plan from `--budget`/`--flops` |
| `--budget` | — | auto-fleet only: $/hr budget for the planner (rejects `--gpu`) |
| `--flops` | — | auto-fleet only: target effective TFLOPs (rejects `--gpu`) |
| `--confirm` | off | auto-fleet only: launch the planned fleet without asking |
| `--model` | — | required: model alias ([docs/MODELS.md](MODELS.md)) or any HF id |
| `--model-kind` | `auto` | `auto`, `causal-lm`, `diffusion`; `auto` infers diffusion for diffusion aliases |
| `--model-revision` | — | HF model branch/tag/commit, resolved to an immutable commit before launch |
| `--data` | — | required (or `--data-mix`): HF dataset id, local path, or object-store URI |
| `--data-revision` | — | HF dataset branch/tag/commit, resolved before launch |
| `--data-format` | `auto` | `auto`, `openai`, `sharegpt`, `alpaca` |
| `--data-mix` | — | alternative to `--data`, RL with `--rl-data-split dispense`: weighted prompt sets, e.g. `a.jsonl=0.7,org/ds@rev=0.3` |
| `--output` | — | object-store URI or `hf://org/repo` (head uploads, then terminates); local or omitted keeps the head up |
| `--trust-remote-code` | off | run code from the pinned model repository |
| `--loss-function` | `cross_entropy` | `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro`, `flow_matching`, or `custom:<file.py>[:<fn>]` |
| `--allow-unsafe-pickled-loss` | off | allow the pickle transport that custom losses need (runs arbitrary code) |

### Launch preflight

Local checks that run before any cloud resource is created.

| Flag | Default | Description |
|---|---|---|
| `--preflight-threads` | `wait` | user thread-count check: `wait` re-checks every 30 s, `error` stops at once, `off` skips |
| `--preflight-thread-start` | `2800` | launch only below this many user threads |
| `--preflight-thread-hard` | `3000` | stop at once at this many threads or more |
| `--preflight-thread-wait-s` | `1800.0` | wait limit for `--preflight-threads wait` |
| `--preflight-memory` | `error` | per-GPU peak memory estimate of each RL island: `error`, `warn`, `off` |
| `--preflight-memory-margin` | `0.9` | allowed peak as a fraction of GPU memory |
| `--rl-island-sample-role` | — | `M:ROLE` per island (`sampler` or `trainer`): sampler islands plus one trainer island over one `--rl-checkpoint-store`; needs `--rl-single-island-no-sync` |
| `--rl-sampler-follow` | `merged` | which model sampler islands load: `merged` or `trainer` |
| `--rl-sampler-spot` | off | sampler islands on spot, the trainer island stays on demand |
| `--rl-island-role` | — | elastic only: per-island `anchor` (on demand) or `droppable` (spot) role |
| `--rl-island-override` | — | negative tests only: give one island a different contract value (needs `--rl-negative-test-run`) |
| `--rl-negative-join-delay-s` | — | negative tests: delay before an overridden island starts (default 180 s) |
| `--rl-negative-test-run` | off | mark the run as a negative test; such runs cannot be resumed or merged as normal runs |

### Fine-tuning

The inner training loop of each island: parameterization, parallelism,
batching and data handling.

| Flag | Default | Description |
|---|---|---|
| `--tuning` | `lora` | `lora` or `full` |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--lora-targets` | `auto` | `auto`, `attention`, `attention-routed-experts`, `all-linear` |
| `--base-quantization` | `none` | `none` or `nf4` (QLoRA; CUDA; needs `--shard ddp`) |
| `--shard` | `ddp` | `ddp` or `fsdp` (shards the frozen base across GPUs/nodes, LoRA only) |
| `--island-backend` | `torch` | `torch` (FSDP2/DDP) or `megatron` (Megatron-Core EP/TP/PP for large MoE) |
| `--expert-parallel` | — | Megatron: EP degree (default fills the island) |
| `--tensor-parallel` | `1` | Megatron: TP degree |
| `--pipeline-parallel` | `1` | Megatron: PP degree |
| `--context-parallel` | `1` | RL (Miles) trainer: CP degree for long sequences |
| `--seq-len` | `2048` | maximum sequence length |
| `--micro-batch-size` | `auto` | per-GPU micro batch; `auto` probes the largest fitting divisor of `--grad-accum` |
| `--grad-accum` | `4` | accumulation steps (with `auto`: per-rank effective sequence batch) |
| `--inner-lr` | `0.0003` | inner (per-island) learning rate |
| `--seed` | `0` | causal-LM root seed |
| `--train-on` | `assistant` | `assistant` (assistant tokens carry loss) or `all` |
| `--assistant-mask-mode` | `native` | `native` (tokenizer's exact assistant mask) or `legacy` (for chat templates without `{% generation %}`) |
| `--attention-backend` | `auto` | `auto`, `sdpa`, `flash-attn-2` |
| `--kernel-backend` | `native` | `native` or `liger` (fused linear cross-entropy) |
| `--max-rows` | — | cap dataset rows per learner |
| `--tokenize` | `stream` | `stream` (async in DataLoader workers) or `preload` |
| `--stream-workers` | `2` | tokenizer worker processes per rank (stream mode) |
| `--resume-from` | — | continue the exact recipe of a local/cloud Yeto LoRA artifact |
| `--branch-from` | — | start a new run from a local/cloud PEFT adapter, recording lineage |
| `--adapter-sha256` | — | required attestation for a cloud `--resume-from`/`--branch-from` artifact |

### Diffusion

Used with `--model-kind diffusion` (or a diffusion alias such as `flux` or
`sd35`). See [docs/DIFFUSION.md](DIFFUSION.md).

| Flag | Default | Description |
|---|---|---|
| `--image-column` | `image` | raw image column |
| `--video-column` | `video` | raw video column |
| `--prompt-column` | `prompt` | prompt column |
| `--height`, `--width` | — | media size override |
| `--resize-mode` | `stretch` | `stretch` or `center-crop` when height and width are set |
| `--num-frames` | — | video frame count for bucketed datasets |
| `--fps` | — | video frame rate for model conditioning |
| `--bucket-by-shape` | off | batch rows by (frames, height, width) |
| `--diffusion-seed` | — | reproducible initialization, data order, timesteps and noise |
| `--diffusion-loss-weighting` | `none` | `none`, `linear`, `sigma`, `snr`, `min-snr` (flow-matching MSE) |
| `--diffusion-min-snr-gamma` | `5.0` | gamma for `min-snr` |
| `--diffusion-adapter` | — | `module:factory` or `file.py:factory` hook for non-standard training steps |
| `--cache-latents` | off | read cached VAE latents from `--latent-column` |
| `--latent-column` | `latents` | cached latent column |
| `--cache-text-embeds` | off | read cached text embeddings from `--text-embeds-column` |
| `--text-embeds-column` | `prompt_embeds` | cached prompt-embedding column |
| `--text-attention-mask-column` | `prompt_attention_mask` | cached text attention-mask column |
| `--pooled-text-embeds-column` | `pooled_prompt_embeds` | cached pooled-embedding column |

### Async sync

The outer (cross-island) optimization: how many sync steps, how fragments are
merged, and how bytes travel over the WAN. See [docs/DESIGN.md](DESIGN.md)
and [docs/PROTOCOL.md](PROTOCOL.md).

| Flag | Default | Description |
|---|---|---|
| `--total-steps` | `64` | outer steps T, one fragment each (RL: sync rounds) |
| `--fragments` | `8` | fragments P (= sync interval H) |
| `--quorum` | `1` | minimum learners per outer step (K) |
| `--sync-interval-steps` | `24.0` | target inner steps per fragment between merges |
| `--pipeline` | `2` | fragment rounds in flight at once (1 = serial); clamped to `--fragments` |
| `--fragment-pattern` | `binpack` | `binpack` (size-balanced) or `strided` (depth-interleaved layers) |
| `--outer-lr` | `0.7` | outer Nesterov learning rate |
| `--outer-momentum` | `0.9` | outer Nesterov momentum |
| `--outer-delayed-nesterov-period` | — | update outer momentum only every N syncs (unset = plain Nesterov) |
| `--delta-correction` | `heloco` | `heloco` (correct stale deltas against outer momentum) or `none` |
| `--merge-alpha` | `0.5` | local weight when applying a broadcast fragment (0 = overwrite) |
| `--grace-ms` | `1000` | cap on the post-quorum grace window |
| `--grace-gamma` | `0.8` | safety margin on the adaptive grace slack |
| `--grace-tau` | `2.0` | compute-overlap budget of the grace window, in inner steps |
| `--external-learners` | `0` | extra slots for machines the launcher does not provision (e.g. a Mac); the log prints the join command |
| `--syncer-shards` | `1` | split the syncer into N processes by fragment range |
| `--syncer-weighted-fold` | `buffered` | `buffered` or `streaming` (running weighted sum) |
| `--wire-dtype` | `bf16` | `bf16`, `f32`, `q4` (4-bit pushes, about 4x less egress) |
| `--wan-streams` | `4` | parallel TCP streams per learner |
| `--wan-stream-schedule` | `rr` | full-parameter RL: `rr` or `adaptive` chunk scheduling |
| `--wan-stall-timeout` | `0.0` | full-parameter RL: redial after this many seconds without progress (0 = off) |
| `--wan-probe-min-mbps` | `0.0` | Miles RL: minimum upload MB/s to the syncer before loading the model (0 = off) |
| `--wan-probe-min-down-mbps` | `0.0` | as above, for the download rate |
| `--wan-probe-mib` | `256` | probe size per direction (MiB) |
| `--wan-probe-relaunches` | `2` | relaunches per island after a rejected link probe |

### Infrastructure

Provisioning: spot vs on-demand, machine shapes, the head/syncer VM, Modal
containers, recovery after preemption, and the RL trainer scheduler. See
[docs/CLOUDS.md](CLOUDS.md).

| Flag | Default | Description |
|---|---|---|
| `--spot` / `--on-demand` | spot | learner capacity type |
| `--cluster-prefix` | `yeto` | cluster name prefix; also the run name |
| `--controller` | `head` | `head` (small on-demand VM hosts syncer and controller) or `local` (this machine must stay up) |
| `--disk-size` | `512` | learner disk (GB) |
| `--learner-cpus` | — | vCPU hint per learner node, e.g. `8+` |
| `--learner-instance-type` | — | pin learner nodes to an exact instance type |
| `--learner-image` | — | machine image: cloud image id, SkyPilot tag, or `docker:` image |
| `--syncer-region` | `us-west-2` | syncer VM placement: AWS region or `cloud/region` |
| `--syncer-memory` | `32` | syncer RAM (GB) |
| `--syncer-instance-type` | — | head VM instance type (default: cheapest meeting `--syncer-memory`) |
| `--syncer-disk-gib` | — | head VM disk (GiB) |
| `--syncer-disk-tier` | — | `low`, `medium`, `high`, `ultra`, `best` |
| `--syncer-public-addr` | — | `HOST:PORT` at which Modal islands reach a syncer with a private address |
| `--keep` | off | do not tear down clusters at the end |
| `--keep-abandoned` | off | keep a learner cluster the controller abandoned |
| `--retry-until-up` | off | keep retrying provisioning until capacity is found |
| `--recover-timeout` | `1200` | seconds to re-provision a failed/preempted learner before continuing without it (0 = no relaunch) |
| `--no-island-relaunch` | off | never relaunch a failed island |
| `--controller-poll` | `30` | fleet controller health poll interval (s) |
| `--spot-without-volume` | off | RL spot islands: no per-island bucket volume for completed rollout groups |
| `--modal-gpu-exact` | off | request the exact Modal GPU type (no H100-to-H200 upgrade) |
| `--modal-gpu-name` | — | fail at start unless every GPU is this model |
| `--modal-retries` | — | Modal function retries after a failed island (Modal runner default 10) |
| `--modal-launcher-relaunch` | off | with `--modal-retries 0`, the launcher still relaunches the island |
| `--modal-timeout-s` | — | Modal function timeout (default 24 h) |
| `--modal-memory-gib` | — | container memory (default 32 GiB per GPU) |
| `--modal-cpu` | — | container CPU cores (default 4 per GPU) |
| `--modal-ephemeral-disk-gib` | — | container disk (GiB) |
| `--modal-env` | — | extra container environment `KEY=VALUE` (repeatable) |
| `--modal-tape-volume` | — | mirror each container's tape files into this Modal Volume |
| `--modal-model-volume` | — | RL: mount this Modal Volume read-only as the model store |
| `--modal-sandbox-secret` | — | Modal Secret for the Modal Sandbox task provider |
| `--rl-trainer-scheduler` | `off` | `off` or `s1` (start/stop the trainer island from the sample-buffer backlog) |
| `--rl-trainer-island` | — | with the scheduler: the island index the scheduler controls |
| `--rl-trainer-scheduler-interval` | `30.0` | seconds between scheduler decisions |
| `--rl-trainer-drain-seconds` | `3600.0` | max time to drain the buffer after the samplers end |
| `--rl-trainer-ready-seconds` | `1800.0` | max time from trainer start to ready |
| `--rl-trainer-max-failures` | `3` | consecutive failed trainer incarnations before the run fails |
| `--rl-trainer-start-seconds` | — | assumed trainer start time (replaced by the measured one) |
| `--rl-trainer-start-seconds-fixed` | off | keep `--rl-trainer-start-seconds` after a measured start |
| `--rl-trainer-command-poll-seconds` | — | command-file poll interval (default 5) |
| `--rl-ssh-trainer-workdir` | — | yeto checkout on the trainer nodes of an `ssh:` trainer island |
| `--rl-node-class` | — | node class per node: `rollout`, `trainer`, `convertible` (optionally `:spot`), repeatable |

### Observability

Weights & Biases telemetry. See [docs/WANDB.md](WANDB.md).

| Flag | Default | Description |
|---|---|---|
| `--wandb` | off | stream training and sync metrics to W&B |
| `--wandb-project` | `yeto` | W&B project |
| `--wandb-entity` | — | W&B team or user |
| `--wandb-mode` | `online` | `online` or `offline` (buffer to disk for a later `wandb sync`) |

### Miles RL (selected)

The Miles RL group has about 290 flags. The tables below list the common
ones by topic; [docs/MILES_RL.md](MILES_RL.md) and `yeto launch --help`
have the complete list. Flags marked "TEST ONLY" in `--help` are for fault
injection and are omitted here.

**Backend and engine**

| Flag | Default | Description |
|---|---|---|
| `--training-mode` | `sft` | `sft` or `rl` |
| `--rl-backend` | `miles` | `miles` or `verl` |
| `--rl-image` | — | digest-pinned RL image `docker:REPO@sha256:DIGEST` (a default is built in) |
| `--rl-model-recipe` | `generic` | model-specific memory/kernel contract: `generic`, `deepseek-v4-flash`, `glm5_next` |
| `--rollout-model` | — | separate inference checkpoint (e.g. FP8) for SGLang; defaults to `--model` |
| `--rollout-num-gpus-per-engine` | `1` | GPUs per colocated SGLang engine |
| `--sglang-mem-fraction-static` | `0.4` | SGLang static memory fraction |
| `--sglang-tp-size`, `--sglang-dp-size`, `--sglang-ep-size` | — | SGLang parallelism |
| `--rl-rollout-quant-tier` | `0` | rollout precision: `0` BF16; `1` FP8 FFN/routed-expert linears (quantized by the engine or the trainer, `--rl-rollout-quant-fp8-source`), KV cache stays BF16; `2` trainer-side FP8 W8A8 releases (needs `--rl-island-sample-role` and `--tuning full`); FP8-capable cards only |
| `--rl-offload-train` | off | offload the trainer between colocated rollout steps (large multi-GPU LoRA) |
| `--rl-activation-recompute` | off | Megatron full activation recompute |
| `--rl-max-tokens-per-gpu` | — | dynamic batching token cap per micro batch |
| `--rl-dp-optimizer` | `auto` | full-parameter DP>1 optimizer: `auto`, `distributed`, `replicated` |
| `--dry-run` | off | validate and print the resource request and learner commands as JSON; creates nothing |

**Data, reward and rollout**

| Flag | Default | Description |
|---|---|---|
| `--reward-function` | — | reward callable `package.module:function` |
| `--rl-prompt-column` | — | prompt column (default: `messages`, `prompt` or `input`) |
| `--rl-label-column` | — | label column passed to the reward (default `label`) |
| `--rl-allow-local-data` | off | accept a local prompt file for `--data` |
| `--n-samples-per-prompt` | `4` | group size |
| `--rollout-batch-size` | `32` | prompts per rollout round |
| `--rollout-max-response-len` | `32768` | max generated tokens per response |
| `--over-sampling-batch-size` | — | oversample prompts for dynamic sampling |
| `--dynamic-sampling-filter-path` | — | Miles group filter (e.g. nonzero reward variance) |
| `--rl-data-split` | `auto` | rows per island: `auto`, `stride`, `dispense`, `shared` |
| `--rl-data-passes` | — | with `dispense`: stop after N passes |
| `--custom-generate-function-path` | — | custom (e.g. agentic) generate function |
| `--agent-max-seq-len` | — | total token cap of one multi-turn agent trajectory |
| `--rl-agentic-harbor` | off | run Harbor trials in sandboxes during rollout |

**Algorithm and optimization**

| Flag | Default | Description |
|---|---|---|
| `--advantage-estimator` | `grpo` | advantage estimator |
| `--rl-algorithm-spec` | — | AlgorithmSpec JSON ([docs/RL_ALGORITHMS.md](RL_ALGORITHMS.md)) |
| `--rl-optimizer-steps` | `1` | optimizer steps per RL round |
| `--rl-lr-schedule` | `auto` | `auto`, `linear`, `constant` |
| `--rl-max-policy-age` | `0` | policy versions a trained sample may lag (0 = on-policy) |
| `--rl-lora-dropout` | — | LoRA dropout during RL training (default 0) |
| `--rl-initial-adapter` | — | local PEFT adapter to start a fresh RL phase from |
| `--rl-deterministic-trainer`, `--deterministic` | off | deterministic trainer kernels |

**Multi-island sync**

| Flag | Default | Description |
|---|---|---|
| `--rl-sync-mode` | `async` | how unset sync options are filled for two or more islands: `async` or `strict` |
| `--rl-sync-preset` | `strict-avg` | `strict-avg`, `decoupled`, `full-parameter`, `sao-streaming-full` |
| `--local-rl-rounds-per-sync` | `1` | local RL rounds between syncs (H) |
| `--rl-sync-full-parameter-rounds` | `sweep` | full-parameter: `sweep` or `async` |
| `--rl-sync-push-dtype` | `f32` | full-parameter push dtype: `f32`, `bf16`, `int8`, `int4`, `int4-log` |
| `--rl-sync-quorum` | — | islands whose pushes close a round |
| `--rl-sync-max-base-lag` | `0` | rounds a push's base may lag |
| `--rl-sync-learner-weight` | `equal` | `equal` or `tokens2-over-steps` |
| `--rl-sync-outer-lr`, `--rl-sync-outer-momentum` | — | outer Nesterov parameters (default from the preset) |

**Placement, multi-node and elastic**

| Flag | Default | Description |
|---|---|---|
| `--rl-placement` | `colocated` | `colocated` or `fixed-partition` |
| `--rl-rollout-gpus` | — | fixed partition: dedicated rollout GPUs per island |
| `--rl-standby-gpus` | `0` | fixed partition: reserved standby GPUs |
| `--rl-min-nodes-per-learner` | `0` | refuse islands with fewer nodes (0 = recipe minimum) |
| `--rl-node-inventory` | — | YAML inventory of SSH-reachable nodes (`ssh:` islands) |
| `--rl-elastic` | off | elastic rollout controller (needs the `--rl-elastic-*` resources/config flags) |
| `--rl-elastic-resources` | — | resources manifest JSON |
| `--rl-elastic-initial-config` | — | initial config id in the manifest |

**Sampler/trainer split and releases**

| Flag | Default | Description |
|---|---|---|
| `--rl-single-island-no-sync` | off | one island, no syncer (also the base of sampler/trainer runs) |
| `--rl-sample-role` | `both` | `both`, `sampler`, `trainer`, `convertible` |
| `--rl-sample-relay` | off | the syncer relays spot sampler groups to the trainer |
| `--rl-spot-subscribers` | `0` | receive-only spot sampler subscribers on the syncer |
| `--rl-release-delta-full-every` | `0` | upload releases as bf16 deltas, with a full release every N |
| `--rl-sample-wait-s` | — | trainer: max wait for one full round from the buffer (default 3600) |

**Checkpoints, recovery and resume**

| Flag | Default | Description |
|---|---|---|
| `--rl-checkpoint-store` | — | off-island state store: bucket URI, `modal-volume://NAME[/PREFIX]`, or a shared path |
| `--rl-cut-every` | — | keep a round cut every N rounds |
| `--rl-cut-auto-mtbf-s` | — | choose the cut interval from an expected MTBF instead |
| `--rl-cut-keep` | — | newest cuts kept (default 2) |
| `--rl-stop-after-rounds` | — | stop after N rounds with a final cut |
| `--rl-resume-allow-config-change` | off | resume a cut although the configuration differs (differences are recorded) |
| `--rl-code-pack` | `auto` | ship code to islands: `auto`, `off`, `git-patch`, `release` |
| `--rl-peer-recovery` / `--no-rl-peer-recovery` | — | full-parameter async: restore a relaunched island from a peer |
| `--rl-export-hf-rounds` | — | with `--rl-checkpoint-store` (full parameter): write each island's model as an HF directory into the store at the given rounds |
| `--rl-stall-timeout` | — | stop the run when no island event arrives for this long (default 900; 0 disables) |

**Evaluation and observation**

| Flag | Default | Description |
|---|---|---|
| `--rl-eval-interval` | — | held-out eval every N rollouts |
| `--rl-eval-data` | — | held-out prompt JSONL |
| `--rl-eval-dataset-name` | — | eval dataset name |
| `--rl-eval-samples-per-prompt` | — | samples per eval prompt |
| `--rl-eval-temperature`, `--rl-eval-top-p` | — | eval sampling (temperature 0 = greedy) |
| `--rl-observe-timeline` | off | record per-round timeline labels |
| `--observe-training-entropy` | off | log training-side policy entropy |
| `--rl-verify-every-round` | off | hash the policy and check engine weights at every publication |

## Running without a cloud

SkyPilot cannot provision on-prem or Huawei Cloud Ascend machines, so
`yeto launch` is not the entry point there. Run the syncer and learners
directly:

```bash
# once, on any machine that can build Rust
cargo build --release --manifest-path syncer/Cargo.toml

# coordinator: 2 islands, quorum 2
./syncer/target/release/yeto-syncer --learners 2 --quorum 2 --total-steps 64 --port 29400

# on each node (one island per node; 8 cards -> 8 ranks)
torchrun --nproc-per-node 8 -m yeto.learner \
  --model lfm25-1b --data org/chat-traces \
  --syncer <syncer-host>:29400 --learner-id 0 --num-learners 2 \
  --shard ddp --tuning lora
```

`--learners` and `--total-steps` are required flags of the `yeto-syncer`
binary; `--quorum` defaults to 1 and `--port` to 29400 (run
`yeto-syncer --help` for the rest); `--syncer`, `--learner-id` and `--num-learners` are
flags of `yeto.learner`. Learners dial out to the syncer, so nodes behind NAT
only need outbound TCP to the syncer's port. The accelerator is detected
automatically (`--device npu` forces Ascend on a mixed node). For tokenizers
whose chat template lacks `{% generation %}` (Qwen3, for example) pass
`--assistant-mask-mode legacy`. See
[docs/ASCEND.md](ASCEND.md#running-a-pure-ascend-fleet).

A Mac joins a cloud fleet into a reserved slot:

```bash
yeto launch --gpu aws:1xa10g@us-west-2 --external-learners 1 --quorum 2 \
  --model lfm25-1b --data org/chat-traces
# then, on the Mac, run the join command printed by the launch log, e.g.
python -m yeto.mlx.learner --model lfm25-1b --data org/chat-traces \
  --syncer <syncer-ip>:29400 --learner-id 1 --num-learners 2
```

See [docs/MLX.md](MLX.md). For Miles RL on existing SSH hosts, see
`yeto-rl-ssh` and [docs/RL_SSH_ACCEPTANCE.md](RL_SSH_ACCEPTANCE.md).
