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

- **`train.py`** — CLI entrypoint. Parses the `--gpu` spec, launches one
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
pip install "yeto[launcher] @ ."      # or: python3 train.py ... from a checkout

yeto \
  --gpu aws:8xa100@us-east-2,aws:8xa100@us-east-1,aws:8xa100@us-west-2 \
  --model deepseek4flash \
  --data armand0e/claude-fable-5-claude-code \
  --loss-function cross_entropy
```

`--gpu` grammar: `cloud:[nodes x]<count>x<gpu>@region`, comma-separated; one
entry per learner. E.g. `aws:4x8xa100@us-east-2` = one learner cluster of
4 nodes × 8×A100 in us-east-2.

Learners run on **spot instances by default** (pass `--on-demand` to opt
out); the syncer VM is always on-demand — it is the cheap, stateful
coordinator, and its checkpoint/resume covers learner preemptions.

## Design notes

- **Merging**: per-learner outer gradient Δ_m,p = Θ_p(prev) − θ_m,p anchored
  at the syncer's previous fragment value; learner weights
  w_m = c_tokens²/c_steps (quantity × quality); weighted RDA per tensor on
  non-embedding fragments, direct averaging on the embedding fragment (whose
  deltas lack the near-orthogonality that motivates RDA).
- **Transport**: custom binary framing over parallel TCP streams (control on
  stream 0; 4 MiB chunks striped across data streams). gRPC was evaluated and
  rejected — protobuf copies and HTTP/2 framing sit on the bulk tensor path.
- **Snapshots**: the single-actor syncer checkpoints at the quiescent cut
  between rounds (params, momentum, per-fragment versions, merged-token
  ledger). `--resume` restores; a JSONL event tape records every merge.
- **Fine-tuning**: `--tuning lora` (default) syncs only adapter weights —
  fragments are megabytes, so the syncer and WAN stay cheap even for large
  models. `--tuning full` syncs everything.
- **Model sizing**: `deepseek4flash` (DeepSeek-V4-Flash, 284B MoE) needs
  ~568 GB for frozen bf16 weights — more than 8×A100-40GB (320 GB); use
  ≥16×80GB GPUs per learner, or pick `gemma4` (12B) / any smaller HF id.
- **Loss masking**: `--train-on assistant` (default) puts loss only on
  assistant-message tokens (plus the closing EOS); `--train-on all` trains
  on every token. Tokenization streams asynchronously in DataLoader workers
  (`--tokenize preload` to materialize upfront).
- **Resilience**: learners reconnect automatically through syncer restarts
  and WAN drops (exponential backoff; work continues locally during the
  outage and re-merges after the post-reconnect rebroadcast). The syncer
  checkpoint is the single durable source of truth — recover a model from it
  with `yeto-export --checkpoint yeto-state.ckpt --model <id> --output-dir out/`
  even if every learner is gone.

## Testing

    python3 -m pytest tests/          # includes a real syncer+learner loop
    (cd syncer && cargo test)
