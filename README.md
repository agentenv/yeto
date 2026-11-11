# Yeto

Yeto post-trains language and diffusion models (SFT and RL) on accelerators
spread across spot instances, regions, clouds and hardware families. Each
learner island trains on its own and exchanges weights with the others through
an asynchronous syncer, so a slow link or a reclaimed spot VM does not stop
the rest of the fleet.

## What it does

**Fine-tuning (SFT)**
- Causal-LM LoRA, QLoRA (NF4 base) and full-parameter fine-tuning;
  diffusion image/video LoRA through Diffusers.
- Asynchronous multi-island training based on Decoupled DiLoCo: islands push
  parameter fragments to a Rust syncer that merges them (quorum with an
  adaptive grace window, token-weighted merge, Nesterov outer step).
- Adapter lifecycle: strict resume (`--resume-from`), branching into a new run
  with recorded lineage (`--branch-from`), and merging an adapter into its base
  model with sharded SafeTensors output (`yeto merge`).

**Reinforcement learning**
- Two trainer backends behind one framework-neutral engine: Miles (default)
  and verl (`--rl-backend`), with Megatron-Core or FSDP inside an island.
- SGLang or vLLM rollout engines, optionally decoupled from the trainer
  through a durable sample buffer (local disk, Modal Volume, S3 or OSS).
- Streaming agentic rollouts that record exact token ids and log-probs.
- Policy releases as full checkpoints or lossless bf16 deltas, checked against
  their sha256.
- Several rollout islands per run with disjoint prompt slots, round cuts and
  checkpoint stores for resume, strict or async multi-island sync
  (`--rl-sync-mode`), and opt-in elastic GPU moves between rollout and
  training (`--rl-elastic`).

**Fleets, clouds and hardware**
- AWS, Nebius, Verda and RunPod through SkyPilot, Modal GPU containers, and
  existing machines over SSH or as external learners.
- A fleet planner (`yeto shape`) that maximizes effective TFLOPs under a $/hr
  budget, or minimizes cost to reach a TFLOPs target.
- NVIDIA GPUs, Huawei Ascend NPUs (910B family) and Apple-silicon Macs (MLX).
- A read-only fleet dashboard and optional Weights & Biases telemetry.

## Architecture

```
                 ┌──────────────────────────────┐
                 │  syncer (Rust, hot path)     │
                 │  fragment ingest · merge     │
                 │  outer step · broadcast      │
                 └──────┬───────┬───────┬───────┘
                        │  TCP (binary framing, WAN)
        ┌───────────────┤               ├───────────────┐
 ┌──────┴──────┐  ┌─────┴───────┐  ┌────┴────────┐
 │ learner 0   │  │ learner 1   │  │ learner 2   │   one island per --gpu entry
 │ cloud A     │  │ cloud B     │  │ Ascend/Mac  │   (PyTorch / Megatron / MLX)
 └─────────────┘  └─────────────┘  └─────────────┘

 RL, decoupled rollout:
 ┌───────────────┐  samples   ┌───────────────┐  batches  ┌──────────────┐
 │ rollout island├───────────►│ sample buffer ├──────────►│ trainer      │
 │ SGLang / vLLM │            │ (disk/object  │           │ (Miles/verl) │
 │ + agent trials│◄───────────┤  store)       │◄──────────┤              │
 └───────────────┘  weight    └───────────────┘  policy   └──────────────┘
                    releases (full or delta)     releases
```

- **Syncer** (`syncer/`, Rust): the hub of the fragment protocol. Every learner
  backend speaks the same protocol and shares one DiLoCo step boundary
  (`yeto/diloco_sync.py`).
- **Learners**: PyTorch FSDP2/DDP (default), Diffusers (`--model-kind
  diffusion`), Megatron-Core (`--island-backend megatron`) and MLX (joined
  with `--external-learners`).
- **RL engine** (`yeto/rl/engine/`): a framework-neutral core that reaches
  backends only through role ports; the adapters live in `yeto/rl/adapters/`.
- **Launch**: the SkyPilot SDK for VM clouds and the Modal SDK for containers.
  By default a small on-demand head VM hosts the syncer and the fleet
  controller (`--controller head`), so the submitting machine can disconnect.
  Learners default to spot; checkpoint/resume absorbs preemptions.

## Install

Python 3.11+. A Rust toolchain (`cargo`) is needed only to run the syncer
yourself or to launch from x86-64 Linux; otherwise the launcher builds it on
the syncer VM.

```bash
git clone <repository-url> yeto && cd yeto
pip install -e ".[launcher]"          # SkyPilot clouds: aws, nebius, verda, runpod
pip install -e ".[launcher,modal]"    # plus Modal containers
```

Other extras: `qlora`, `diffusion`, `mlx`, `wandb`, `local-rl`,
`a100-liger`, `a100-flash-attn`, `dev`, `test` (see `pyproject.toml`).

Put each cloud's credentials where its own CLI keeps them, run `sky check`, and
export `HF_TOKEN` for gated models, private data and `hf://` outputs (see
[docs/CLOUDS.md](docs/CLOUDS.md)).

## Quick start

Launches are detached; the run name is `--cluster-prefix` (default `yeto`).

**SFT on a fleet you choose** (one island per `--gpu` entry,
`cloud:[NODESx]COUNTxGPU[@region]`):

```bash
yeto launch --gpu aws:8xa100@us-east-2,nebius:8xh100@eu-north1 \
  --model qwen35-9b --data org/chat-traces \
  --output s3://my-bucket/qwen35-9b-sft --cluster-prefix qwen-sft
yeto logs qwen-sft
```

**Let the planner pick the fleet** under a $/hr budget:

```bash
yeto shape --model qwen35-9b --budget 40                              # print the plan
yeto shape --model qwen35-9b --budget 40 --data org/chat-traces --apply   # and launch it
```

**RL with the Miles backend** (add `--dry-run` to validate without creating
resources):

```bash
yeto launch --training-mode rl --gpu aws:8xa100@us-east-1,aws:8xa100@us-west-2 \
  --model Qwen/Qwen3-1.7B --model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --data HuggingFaceH4/MATH-500 --data-revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
  --rl-prompt-column problem --rl-label-column answer \
  --reward-function yeto.rl.math_reward:reward_func --trust-remote-code \
  --tuning lora --lora-targets attention --inner-lr 1e-5 \
  --rollout-batch-size 16 --n-samples-per-prompt 4 --total-steps 8
```

RL needs immutable inputs: pass the 40-character Hub commits with
`--model-revision` / `--data-revision`. `--reward-function` takes a built-in
reward (`yeto.rl.*`, listed in [examples/README.md](examples/README.md)), any
`package.module:function` importable from the repository root or the
environment, or a `path/to/file.py:function` inside the repository.

The `--gpu` grammar, data formats, `--output`, QLoRA, resume/branch and every
flag are described in [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md).

## Commands

| Command | Purpose |
|---|---|
| `yeto launch` | submit a detached SFT or RL run |
| `yeto shape` | plan a fleet for a model under a budget or TFLOPs target (`--apply` launches it) |
| `yeto status` / `logs` / `down` / `stop-run` | list runs, follow a log, tear down, stop cleanly |
| `yeto merge` | merge a PEFT adapter into its base and write SafeTensors shards |
| `yeto sample-diffusion` | sample from a diffusion adapter on a cloud GPU |
| `yeto dashboard serve` / `export` / `mirror` | read-only RL fleet dashboard |
| `yeto rl` | experimental single-process CyberGym PPO |
| `yeto-export`, `yeto-rl-export`, `yeto-diffusion-export` | export syncer or RL checkpoints as HF/PEFT/Diffusers artifacts |
| `yeto-diffusion-sample` | sample from a diffusion adapter locally |
| `yeto-rl-ssh`, `yeto-stage1` | RL on existing SSH hosts |
| `yeto-codex-traces` | convert Codex traces to chat JSONL |
| `yeto-wandb-tape` | upload a syncer event tape to W&B |

Without a cloud (on-prem Ascend, a Mac joining a fleet), run the syncer and
learners directly; see
[docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md#running-without-a-cloud).

## Documentation

| Doc | Topic |
|---|---|
| [CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | every command and flag, with defaults and examples |
| [MODELS.md](docs/MODELS.md) | model aliases and their memory footprint |
| [DESIGN.md](docs/DESIGN.md) | merge math, blending, adaptive grace, wire formats, resilience |
| [PROTOCOL.md](docs/PROTOCOL.md) | learner/syncer wire protocol, RL sync modes, island pool |
| [MILES_RL.md](docs/MILES_RL.md) | Miles RL: rollout/training boundaries, recovery, export, all RL flags |
| [RL_ALGORITHMS.md](docs/RL_ALGORITHMS.md) | RL algorithm specs |
| [CLOUDS.md](docs/CLOUDS.md) | per-cloud credentials, regions, planner signals, Modal islands |
| [KIMI_K31_ASYNC_RL_GUIDE.md](docs/KIMI_K31_ASYNC_RL_GUIDE.md) | Kimi K3.1 async GRPO (agentic RL) end to end: Modal trainer, NC samplers, run config, dashboard, operations |
| [K31_FULL_262K_LAYOUT.md](docs/K31_FULL_262K_LAYOUT.md) | Kimi K3.1 128-expert full-parameter trainer layout on 6x8 B300 |
| [NC_POOL.md](docs/NC_POOL.md) | National Compute sampler pool: config, dry run, status, stop, billing |
| [DISPENSER.md](docs/DISPENSER.md) | task dispenser over a shared bucket (store dispenser) |
| [ASCEND.md](docs/ASCEND.md) | Ascend NPU islands and a pure-Ascend fleet |
| [MEGATRON.md](docs/MEGATRON.md) | Megatron-Core island backend for large MoE |
| [MLX.md](docs/MLX.md) | Apple-silicon islands |
| [DIFFUSION.md](docs/DIFFUSION.md) | Diffusers image/video backend |
| [ADAPTER_LIFECYCLE.md](docs/ADAPTER_LIFECYCLE.md) | LoRA resume, branching, merge, export |
| [PROVENANCE.md](docs/PROVENANCE.md) | source pinning, attestation, artifact provenance |
| [DASHBOARD.md](docs/DASHBOARD.md) | RL fleet dashboard |
| [WANDB.md](docs/WANDB.md) | W&B telemetry |
| [RL_SSH_ACCEPTANCE.md](docs/RL_SSH_ACCEPTANCE.md) | RL on existing SSH hosts |
| [CYBERGYM_RL.md](docs/CYBERGYM_RL.md) | experimental local PPO with CyberGym rewards |
| [A100_KERNELS.md](docs/A100_KERNELS.md) | opt-in attention and loss kernels |
| [LM_BENCHMARK.md](docs/LM_BENCHMARK.md), [DIFFUSION_BENCHMARK.md](docs/DIFFUSION_BENCHMARK.md), [RL_BENCHMARK.md](docs/RL_BENCHMARK.md), [RL_ELASTIC_BENCHMARK.md](docs/RL_ELASTIC_BENCHMARK.md) | benchmark contracts and runners |
| [docker/README.md](docker/README.md) | engine images and version locks |
| [tutorial/hello-world](tutorial/hello-world/README.md) | end-to-end SFT tutorial on Codex traces |
| [examples/README.md](examples/README.md) | reward and algorithm examples |
