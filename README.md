# Yeto

**Yeto** fine-tunes language models across cheap, geographically scattered
GPU capacity — spot instances, mixed regions, mixed clouds — via the
[SkyPilot](https://skypilot.co) SDK. Its asynchronous synchronization is
based on **Decoupled DiLoCo**
([Douillard et al., arXiv 2604.21428](https://arxiv.org/abs/2604.21428)):
a Rust syncer merges parameter fragments from independent learner islands
(quorum + adaptive grace, token-weighted RDA, Nesterov outer step), so slow
links and preempted islands never block training.

```
                 ┌──────────────────────────────┐
                 │  syncer (Rust, hot path)     │
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

## Usage

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
  dir), or cloud URI (`s3://`, `gs://`, `r2://`, …) — non-HF sources ship to
  learners via SkyPilot file mounts.
- `--model`: any HF id (private repos work with `HF_TOKEN`), or an alias
  below.
- Learners default to spot; the head VM (syncer + fleet controller) is a
  small on-demand box whose checkpoint/resume absorbs preemptions. The
  submitting machine can disconnect after launch.

## Supported models

Aliases are sugar over `yeto/models.py` (this table is generated from it;
a test keeps them in sync). "bf16 GB" is the frozen-base footprint used for
island/disk sizing; "(Hub)" resolves from safetensors metadata at plan time.

| alias | Hugging Face id | bf16 GB |
|---|---|---|
| `gemma4` | `google/gemma-4-12B-it` | 66 |
| `deepseek4flash` | `deepseek-ai/DeepSeek-V4-Flash` | 568 |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 17 |
| `qwen35-4b` | `Qwen/Qwen3.5-4B` | 8 |
| `qwen35-9b` | `Qwen/Qwen3.5-9B` | 18 |
| `qwen35-9b-base` | `Qwen/Qwen3.5-9B-Base` | 18 |
| `qwen35-35b-a3b` | `Qwen/Qwen3.5-35B-A3B-Base` | 70 |
| `qwen35-397b-a17b` | `Qwen/Qwen3.5-397B-A17B` | 794 |
| `qwen36-27b` | `Qwen/Qwen3.6-27B` | 54 |
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
| `glm52` | `zai-org/GLM-5.2` | 1488 |
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
| `ornith-397b` | `deepreinforce-ai/Ornith-1.0-397B-FP8` | 794 |
| `lfm25-230m` | `LiquidAI/LFM2.5-230M` | 0.5 |
| `lfm25-1b` | `LiquidAI/LFM2.5-1B` | 2 |
| `lfm25-8b-a1b` | `LiquidAI/LFM2.5-8B-A1B` | 17 |
| `vibethinker-15b` | `WeiboAI/VibeThinker-1.5B` | 3 |
| `vibethinker-3b` | `WeiboAI/VibeThinker-3B` | 6 |

## Docs

[docs/DESIGN.md](docs/DESIGN.md) — merge math, blending, adaptive grace,
delta correction, q4 wire format, snapshots, resilience.
[docs/PROTOCOL.md](docs/PROTOCOL.md) — the learner↔syncer wire protocol.

## Testing

    python3 -m pytest tests/          # includes a real syncer+learner loop
    (cd syncer && cargo test)
