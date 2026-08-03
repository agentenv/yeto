# Yeto

**Yeto** fine-tunes language and diffusion models across cheap, geographically
scattered GPU capacity — spot instances, mixed regions, mixed clouds, even
mixed hardware families.

```
                 ┌──────────────────────────────┐
                 │  syncer (hot path)           │
                 │  fragment ingest · RDA merge │
                 │  Nesterov outer step · bcast │
                 └──────┬───────┬───────┬───────┘
                        │  TCP (binary framing, WAN)
        ┌───────────────┤               ├───────────────┐
 ┌──────┴──────┐  ┌─────┴───────┐  ┌────┴────────┐
 │ learner 0   │  │ learner 1   │  │ learner 2   │   … one island per --gpu entry
 │ us-east-2   │  │ us-east-1   │  │ runpod:CA   │   (PyTorch, AdamW inner opt)
 └─────────────┘  └─────────────┘  └─────────────┘
```

## Quick start

```bash
pip install "yeto[launcher] @ ."

# pick the fleet yourself…
yeto launch --gpu aws:8xa100@us-east-2,runpod:8xh100@CA \
  --model qwen35-9b --data org/chat-traces

# …or let the planner pick it (budget $/hr and/or a TFLOPs target)
yeto launch --model qwen35-9b --data org/chat-traces --budget 40 --confirm

yeto status | logs <run> | down <run>   # runs detach; Ctrl-C never kills them
```

- `--gpu` grammar: `cloud:[nodes x]<count>x<gpu>[@region]`, one entry per
  learner island.
- Omitting `--gpu` invokes `yeto shape`: an exact solver maximizes effective
  TFLOPs under your budget (or minimizes cost to reach `--flops`), subject to
  live spot quotas minus usage, spot placement scores, RunPod stock, and an
  FSDP memory model of the model. Run `yeto shape` directly to see the plan,
  rejected shapes with reasons, and the launch line without launching.
- `--data`: HF dataset id, local path (jsonl/json/parquet or `save_to_disk`
  dir), or any sky-supported object-store URI — non-HF sources ship to
  learners via SkyPilot file mounts.
- `--data-format`: normalize OpenAI `messages`, ShareGPT `conversations`, or
  Alpaca `instruction`/`input`/`output` rows into the same chat representation
  before tokenization. The default, `auto`, detects the schema per row and
  reports ambiguous or malformed rows with their row number.
- QLoRA uses `--tuning lora --base-quantization nf4 --shard ddp`. It stores the
  frozen base in bitsandbytes NF4 with double quantization and bf16 compute;
  pass `--gpu` explicitly while the fleet planner's QLoRA memory model is being
  calibrated.
- Existing causal LoRA artifacts can be continued with `--resume-from` when
  the recorded model, data, and recipe match, or used as a new lineage with
  `--branch-from`. `yeto merge --max-shard-size 2GB ...` safely folds an
  adapter into its base and writes deployment-ready SafeTensors shards. See
  [docs/ADAPTER_LIFECYCLE.md](docs/ADAPTER_LIFECYCLE.md).
- A local CyberGym server can provide execution-grounded rewards to the
  experimental `yeto rl` loop. See
  [docs/CYBERGYM_RL.md](docs/CYBERGYM_RL.md) for the safe local setup, smoke
  command, test evidence, and current limitations.
- Distributed RL uses `yeto launch --training-mode rl` to run a fixed roster
  of pinned Miles islands with strict synchronous LoRA averaging. See
  [docs/TEMP_Decoupled_DiloCo_RL_Support.md](docs/TEMP_Decoupled_DiloCo_RL_Support.md)
  for the runtime contract, launch configuration, recovery model, and
  validation evidence. For two already-provisioned H200s reachable over
  SSH/Tailscale, use the acceptance harness in
  [docs/RL_SSH_ACCEPTANCE.md](docs/RL_SSH_ACCEPTANCE.md).
- `--output`: any sky-supported store URI or `hf://org/repo` — the head
  fetches the model from the winning learner, uploads it, and **terminates
  itself** (fully self-cleaning run). Local path or omitted: the artifact
  stays on the head and the head is kept up.
- `--model`: any HF id (private repos work with `HF_TOKEN`), or an alias
  below.
- Learners default to spot; the head VM (syncer + fleet controller) is a
  small on-demand box whose checkpoint/resume absorbs preemptions. The
  submitting machine can disconnect after launch.

## Architecture

Fleets launch via the [SkyPilot](https://skypilot.co) SDK; asynchronous
synchronization is based on **Decoupled DiLoCo**
([Douillard et al., arXiv 2604.21428](https://arxiv.org/abs/2604.21428)):
the syncer merges parameter fragments from independent learner islands
(quorum + adaptive grace, token-weighted RDA, Nesterov outer step), so slow
links and preempted islands never block training.

One protocol, one syncer, four learner backends. Every backend speaks the
same fragment protocol and runs the same DiLoCo step boundary — the
pull/merge/α-blend/push loop lives in one shared module
(`yeto/diloco_sync.py`), so protocol changes land once and apply everywhere.

| backend | selector | scope |
|---|---|---|
| PyTorch (FSDP2/DDP) | default | causal LM, LoRA/full |
| Diffusers | `--model-kind diffusion` | image/video LoRA |
| Megatron-Core | `--island-backend megatron` | EP islands for 1T-class MoE |
| MLX | `--external-learners` | Apple-silicon islands |

Hardware families are isolated behind `yeto/accel.py` (CUDA / Ascend NPU
policy functions) rather than `device.type` branches in shared code. The
terminal contract — budget cutoff, authoritative final cut, checkpoint
marking — is shared by all backends (`finalization.py`,
`budget_finalization.py`, `final_marker.py`).

## Supported models

Aliases are sugar over `yeto/models.py` (this table is generated by
`scripts/gen_model_table.py`; a test keeps them in sync). "Min island VRAM"
is the frozen-base footprint an island's GPUs must jointly hold (bf16 base,
LoRA) — add ~8 GB per GPU for activations/overhead, ×8 for full tuning;
"(Hub)" means the size is resolved from safetensors metadata at plan time.

Tested — a completed Yeto fine-tuning run on real hardware:

| alias | Hugging Face id | min island VRAM (GB) |
|---|---|---|
| `deepseek4flash` | `deepseek-ai/DeepSeek-V4-Flash` | 568 |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | (Hub) |
| `qwen36-27b` | `Qwen/Qwen3.6-27B` | 54 |
| `kimi-k3` | `moonshotai/Kimi-K3` | (Hub) |
| `glm52` | `zai-org/GLM-5.2` | 1488 |
| `laguna-s-2.1` | `poolside/Laguna-S-2.1` | (Hub) |

<details>
<summary>All other supported aliases (untested)</summary>

| alias | Hugging Face id | min island VRAM (GB) |
|---|---|---|
| `gemma4` | `google/gemma-4-12B-it` | 66 |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 17 |
| `qwen35-4b` | `Qwen/Qwen3.5-4B` | 8 |
| `qwen35-9b` | `Qwen/Qwen3.5-9B` | 18 |
| `qwen35-9b-base` | `Qwen/Qwen3.5-9B-Base` | 18 |
| `qwen35-35b-a3b` | `Qwen/Qwen3.5-35B-A3B-Base` | 70 |
| `qwen35-397b-a17b` | `Qwen/Qwen3.5-397B-A17B` | 794 |
| `qwen36-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | 70 |
| `llama32-1b` | `meta-llama/Llama-3.2-1B` | 3 |
| `llama32-3b` | `meta-llama/Llama-3.2-3B` | 7 |
| `llama31-8b` | `meta-llama/Llama-3.1-8B` | 16 |
| `llama31-8b-it` | `meta-llama/Llama-3.1-8B-Instruct` | 16 |
| `llama31-70b` | `meta-llama/Llama-3.1-70B` | 141 |
| `llama33-70b-it` | `meta-llama/Llama-3.3-70B-Instruct` | 141 |
| `gptoss-20b` | `openai/gpt-oss-20b` | 42 |
| `gptoss-120b` | `openai/gpt-oss-120b` | 234 |
| `kimi-k2-thinking` | `moonshotai/Kimi-K2-Thinking` | 2060 |
| `kimi-k25` | `moonshotai/Kimi-K2.5` | 2060 |
| `kimi-k26` | `moonshotai/Kimi-K2.6` | 2060 |
| `deepseek31` | `deepseek-ai/DeepSeek-V3.1` | 1343 |
| `deepseek-r1` | `deepseek-ai/DeepSeek-R1` | 1343 |
| `deepseek4pro` | `deepseek-ai/DeepSeek-V4-Pro` | 3200 |
| `nemotron3-nano` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 61 |
| `nemotron3-super` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | 240 |
| `nemotron3-ultra` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | 1100 |
| `glm45-air` | `zai-org/GLM-4.5-Air` | 212 |
| `glm46` | `zai-org/GLM-4.6` | 714 |
| `llama4-scout` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 218 |
| `llama4-maverick` | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | 800 |
| `qwen3-coder-480b` | `Qwen/Qwen3-Coder-480B-A35B-Instruct` | 960 |
| `minimax-m2` | `MiniMaxAI/MiniMax-M2` | 460 |
| `minimax-m3` | `MiniMaxAI/MiniMax-M3` | (Hub) |
| `kimi-k27-code` | `moonshotai/Kimi-K2.7-Code` | (Hub) |
| `mistral-small3` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 48 |
| `ornith-9b` | `deepreinforce-ai/Ornith-1.0-9B` | 19 |
| `ornith-31b` | `deepreinforce-ai/Ornith-1.0-31B` | 62 |
| `ornith-35b` | `deepreinforce-ai/Ornith-1.0-35B` | 70 |
| `ornith-397b` | `deepreinforce-ai/Ornith-1.0-397B` | 794 |
| `lfm25-230m` | `LiquidAI/LFM2.5-230M` | 0.5 |
| `lfm25-1b` | `LiquidAI/LFM2.5-1.2B-Instruct` | 3 |
| `lfm25-8b-a1b` | `LiquidAI/LFM2.5-8B-A1B` | 17 |
| `vibethinker-15b` | `WeiboAI/VibeThinker-1.5B` | 3 |
| `vibethinker-3b` | `WeiboAI/VibeThinker-3B` | 6 |
| `deepseek4flash-bf16` | `RedHatAI/DeepSeek-V4-Flash-BF16` | 568 |
| `deepseek31-bf16` | `unsloth/DeepSeek-V3.1-BF16` | 1343 |
| `deepseek-r1-bf16` | `unsloth/DeepSeek-R1-BF16` | 1343 |
| `gptoss-20b-bf16` | `axolotl-ai-co/gpt-oss-20b-dequantized` | 42 |
| `gptoss-120b-bf16` | `axolotl-ai-co/gpt-oss-120b-dequantized` | 234 |
| `kimi-k2-thinking-bf16` | `unsloth/Kimi-K2-Thinking-BF16` | 2060 |
| `deepseek31-base-bf16` | `unsloth/DeepSeek-V3.1-Base-BF16` | 1343 |
| `kimi-k2-base-bf16` | `unsloth/Kimi-K2-Base-BF16` | 2060 |

</details>

## Docs

[docs/DESIGN.md](docs/DESIGN.md) — merge math, blending, adaptive grace,
delta correction, q4 wire format, snapshots, resilience.
[docs/PROTOCOL.md](docs/PROTOCOL.md) — the learner↔syncer wire protocol.
[docs/PROVENANCE.md](docs/PROVENANCE.md) — source pinning, attestation, and
artifact provenance.
[docs/CYBERGYM_RL.md](docs/CYBERGYM_RL.md) — local CyberGym setup and RL
smoke-run guide.
[docs/TEMP_Decoupled_DiloCo_RL_Support.md](docs/TEMP_Decoupled_DiloCo_RL_Support.md)
— strict synchronous Miles RL architecture, launch contract, and validation.
[docs/RL_SSH_ACCEPTANCE.md](docs/RL_SSH_ACCEPTANCE.md) — run and independently
verify strict RL on two existing H200 hosts over SSH/Tailscale.
[docs/ADAPTER_LIFECYCLE.md](docs/ADAPTER_LIFECYCLE.md) — strict adapter
resume, intentional branching, safe base-model merge, and export sharding.
[docs/DIFFUSION.md](docs/DIFFUSION.md) — the generic Diffusers image/video
backend, data and conditioning contracts, external adapters, export, sampling,
validation, and current limitations.
[docs/MEGATRON.md](docs/MEGATRON.md) — the Megatron-Core island backend (EP
for 1T-class MoE; runs inside the NGC NeMo container).
[docs/MLX.md](docs/MLX.md) — the Apple-silicon island backend: Macs as
learner islands (`yeto launch --external-learners`, cross Mac↔NVIDIA runs).
[docs/ASCEND.md](docs/ASCEND.md) — Ascend NPU islands: the accelerator
abstraction, the refused CUDA-only paths, and running a pure-Ascend fleet.
[docs/A100_KERNELS.md](docs/A100_KERNELS.md) — opt-in causal attention/model
kernels, correctness gates, pinned dependencies, and the standalone 8xA100
throughput and memory benchmark.
[docs/LM_BENCHMARK.md](docs/LM_BENCHMARK.md) — standalone equal-hardware
causal-LM benchmark contract, workload controls, complete arm and metric
tables, and reproducibility rules.
[docs/DIFFUSION_BENCHMARK.md](docs/DIFFUSION_BENCHMARK.md) — standalone
equal-hardware diffusion benchmark contract, media controls, complete arm and
metric tables, and reproducibility rules.
[docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) — aggregate and
per-seed results for the completed Qwen3.6, LTX-Video, and Wan2.2 benchmarks.

## Testing and CI

    python3 -m pytest tests/          # includes a real syncer+learner loop
    (cd syncer && cargo test)

CI runs three jobs on every PR and push to main: the Rust syncer suite, the
Python suite on CPU, and a **GPU smoke test** on a self-hosted runner
(`[self-hosted, gpu]` labels) — the real syncer plus two real learners
training SmolLM2-135M on CUDA, with event-tape assertions that every outer
step merged pushed deltas from both learners.

Three heavier harnesses (all support `--dry-run`):

    # smoke every supported model with the auto fleet planner, tiered by
    # size; sequential, self-cleaning, writes a pass/fail report
    python scripts/smoke_models.py --tier small --dry-run

    # causal-LM quality check against equal-hardware synchronous baselines
    python scripts/compare_diloco.py --data <chat.jsonl> --settings all --dry-run

    # diffusion quality check with an explicit, fixed media shape
    python scripts/benchmark_diffusion_diloco.py \
        --model Lightricks/LTX-Video --data <media-dataset> \
        --height 512 --width 512 --settings all --dry-run
