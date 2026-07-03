# Yeto

**Yeto** is an efficient, low-cost post-training tool: it fine-tunes language
models across cheap, geographically scattered GPU clusters (spot instances,
mixed regions, mixed clouds) launched with the [SkyPilot](https://skypilot.co)
SDK.

Yeto's asynchronous synchronization algorithm is based on **Decoupled DiLoCo**
([Douillard et al., arXiv 2604.21428](https://arxiv.org/abs/2604.21428)).

## Architecture

```
                 ┌──────────────────────────────┐
                 │  syncer (Rust, hot path)     │
                 │  fragment ingest · RDA merge │
                 │  Nesterov outer step · bcast │
                 └──────┬───────┬───────┬───────┘
                        │  TCP (binary framing, WAN)
        ┌───────────────┤               ├───────────────┐
 ┌──────┴──────┐  ┌─────┴───────┐  ┌────┴────────┐
 │ learner 0   │  │ learner 1   │  │ learner 2   │   … one per --gpu entry
 │ us-east-2   │  │ us-east-1   │  │ us-west-2   │
 │ 8×A100 node │  │ 8×A100 node │  │ 8×A100 node │   (PyTorch, AdamW inner opt)
 └─────────────┘  └─────────────┘  └─────────────┘
```

- **`yeto` CLI** — parses the `--gpu` spec, launches one
  SkyPilot cluster per learner plus a syncer VM, wires up IPs/ports, streams logs.
- **`syncer/`** — Rust implementation of the latency-sensitive syncer:
  async TCP server, per-fragment sync schedule (interval `H`, round-robin
  offsets), quorum-`K` gather with grace window, token/step-weighted RDA
  merge, Nesterov outer optimizer, consistent checkpoints, JSONL event tape.
- **`yeto/`** — Python package: learner training loop (HF transformers +
  AdamW inner steps, background fragment push/pull), data loading, loss
  functions, GPU-spec parsing, SkyPilot orchestration.

## Usage

```bash
pip install "yeto[launcher] @ ."

yeto launch \
  --gpu aws:8xa100@us-east-2,aws:8xa100@us-east-1,aws:8xa100@us-west-2 \
  --model deepseek4flash \
  --data armand0e/claude-fable-5-claude-code \
  --loss-function cross_entropy
```

`launch` provisions a small on-demand **head VM** that hosts both the
syncer and the fleet controller (SkyPilot managed-jobs style): the
submitting machine is free to disconnect the moment submission finishes,
and **Ctrl-C merely detaches** from the log stream. Runs are named by
`--cluster-prefix` (default `yeto`). `--controller local` keeps the
controller on your host instead. The head VM stays up after the run
until you `yeto down <run>`.

```bash
yeto status                # table of known runs
yeto logs <run>            # re-attach to a run's log stream (--no-follow to dump)
yeto down <run>            # stop the run's worker and tear down its clusters
```

`--gpu` grammar: `cloud:[nodes x]<count>x<gpu>@region`, comma-separated; one
entry per learner. E.g. `aws:4x8xa100@us-east-2` = one learner cluster of
4 nodes × 8×A100 in us-east-2.

Don't want to pick the fleet yourself? `yeto shape` computes it:

```console
yeto shape --model gemma4 --budget 40 --data <hf-dataset> [--apply]
```

maximizes effective training FLOPs under your $/hr budget, your account's
remaining spot quotas (limit minus what is already running), and spot
placement scores (> 7 by default), then prints the matching `yeto launch`
line — `--apply` runs it. Island sizes come from an FSDP memory model of the
model's bf16 footprint (fp8/fp32 checkpoints are normalized). Budgets are
enforced with a spot-price margin (`--price-margin`, default 15%) because
catalog prices are estimates. Signals are fetched in one parallel wave and
cached for an hour; rejected candidates are listed with reasons, `--json`
emits a machine-readable plan, `--regions all` searches every catalog
region. With RunPod credentials present, RunPod pods join the candidate
pool — no quotas there, so live per-GPU stock levels gate and cap those
shapes instead (`--clouds` to control). Unfetchable capacity signals are
assumed best-case with a warning; `--strict-capacity-check` rejects them
and `--skip-capacity-check` plans on quota + price alone.

Learners run on **spot instances by default** (pass `--on-demand` to opt
out); the syncer VM is always on-demand — it is the cheap, stateful
coordinator, and its checkpoint/resume covers learner preemptions.

## Supported models

Any Hugging Face model id works with `--model`; the aliases below are
sugar (single source: `yeto/models.py` — this table is generated from it
and a test keeps them in sync). "bf16 GB" is the frozen-base footprint the
planner sizes islands and disks from; "(Hub)" means the size is resolved
from safetensors metadata at plan time.

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

## Design notes

See [docs/DESIGN.md](docs/DESIGN.md) for the merge math, transport,
q4 wire format, snapshots, and resilience notes, and
[docs/PROTOCOL.md](docs/PROTOCOL.md) for the wire protocol.

## Testing

    python3 -m pytest tests/          # includes a real syncer+learner loop
    (cd syncer && cargo test)
