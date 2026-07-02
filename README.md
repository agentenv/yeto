# Decoupled DiLoCo on SkyPilot

Multi-region fine-tuning using **Decoupled DiLoCo** ([arXiv 2604.21428](https://arxiv.org/abs/2604.21428)),
launched across clouds/regions with the [SkyPilot](https://skypilot.co) Python SDK.

Decoupled DiLoCo decomposes training into `M` independent **learners** (one per
region/cluster) coordinated by a central **syncer**. Learners run inner
optimization (AdamW) on their data shard and never block on peers; the syncer
asynchronously pulls per-fragment pseudo-gradients from a quorum of `K ≤ M`
learners, merges them (weighted radial-directional averaging), applies an outer
Nesterov step, and broadcasts fragments back.

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

- **`train.py`** — CLI entrypoint. Parses the `--gpu` spec, launches one
  SkyPilot cluster per learner plus a syncer VM, wires up IPs/ports, streams logs.
- **`syncer/`** — Rust implementation of the latency-sensitive syncer:
  async TCP server, per-fragment sync schedule (interval `H`, offsets `t_p`),
  quorum-`K` gather with grace window, token/step-weighted RDA merge,
  Nesterov outer optimizer.
- **`decoupled_diloco/`** — Python package: learner training loop
  (HF transformers + AdamW inner steps, background fragment push/pull),
  data loading, loss functions, GPU-spec parsing, SkyPilot
  orchestration.

## Usage

```bash
python3 train.py \
  --gpu aws:8xa100@us-east-2,aws:8xa100@us-east-1,aws:8xa100@us-west-2 \
  --model deepseek4flash \
  --data armand0e/claude-fable-5-claude-code \
  --loss-function cross_entropy
```

`--gpu` grammar: `cloud:[nodes x]<count>x<gpu>@region`, comma-separated; one
entry per learner. E.g. `aws:4x8xa100@us-east-2` = one learner cluster of
4 nodes × 8×A100 in us-east-2.

## Status

Work in progress — see git history.
